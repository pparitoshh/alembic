"""Stage 1: document-grounded question generation + multiple grounded answers per question.

Resumable: questions and answers are appended to the run dir as they finish, and a rerun only
does the missing ones. On a shared cluster node, a job killed at its wall time loses nothing.
"""

import itertools
import random
from pathlib import Path

from .config import Config
from .io import JsonlAppender
from .records import prose_row
from .schemas import GeneratedQuestion
from .seeds import load_chunks
from .teacher import Teacher

Q_SYSTEM = 'You write realistic questions that users of an HPC cluster ask. Reply only with JSON: {"question": "<the question>"}.'

Q_PROMPT = """Reference documentation:
<doc>
{chunk}
</doc>

Write ONE question that {persona} might ask, of type "{task}":
- concept: ask to explain a concept, option or difference
- howto: ask how to achieve something concrete
- script: ask for a complete sbatch/shell script with specific requirements
- debug: describe a concrete problem (error message, pending reason, failed job) and ask why/how to fix

The question must be answerable from the documentation above, but must not mention "the documentation".
Write it in the persona's own words. Reply only with JSON: {{"question": "<the question>"}}."""

A_SYSTEM = """You are an expert HPC assistant. Answer concisely and correctly.
- Keep it short and direct: a few sentences, plus at most one script or command block.
- Put scripts and commands in fenced code blocks.
- Only use options and commands you are sure exist.
- If the answer depends on the cluster configuration, say what you assume.
- Never mention "the documentation" or "the reference"; answer directly."""

A_PROMPT = """Use this reference material as ground truth:
<doc>
{chunk}
</doc>

Question: {question}"""


def question_jobs(cfg: Config) -> list[dict]:
    """Deterministic (chunk, persona, task) grid sample; `id` is stable across reruns of the same config."""
    gcfg = cfg.generate
    train_chunks, _ = load_chunks(cfg)
    rng = random.Random(0)
    grid = list(itertools.product(range(len(gcfg.personas)), gcfg.task_types))
    jobs = []
    for chunk in train_chunks:
        for p, task in rng.sample(grid, min(gcfg.questions_per_chunk, len(grid))):
            jobs.append({**chunk, "id": f"{chunk['chunk_id']}/{task}/p{p}", "persona": gcfg.personas[p], "task": task})
    return jobs


def run(cfg: Config) -> Path:
    gcfg, run_dir = cfg.generate, cfg.run_dir
    teacher = Teacher(cfg.teacher)
    jobs = question_jobs(cfg)

    q_out = JsonlAppender(run_dir / "questions.jsonl")
    done_q = {q["id"]: q for q in q_out.existing()}
    todo_q = [j for j in jobs if j["id"] not in done_q]
    print(f"[generate] {len(jobs)} questions planned, {len(done_q)} already done, {len(todo_q)} to go")

    def make_question(j):
        q, _ = teacher.chat_json(Q_SYSTEM, Q_PROMPT.format(chunk=j["text"], persona=j["persona"], task=j["task"]), GeneratedQuestion)
        if q:
            q_out.append({**j, "question": q.question.strip()})

    teacher.map(make_question, todo_q)
    questions = [q for q in q_out.existing() if q["id"] in {j["id"] for j in jobs}]
    if len(questions) < len(jobs):
        print(f"[generate] {len(jobs) - len(questions)} questions had invalid JSON; rerun to retry them")

    out = JsonlAppender(run_dir / "generated.jsonl")
    lp_out = JsonlAppender(run_dir / "teacher_logprobs.jsonl.gz") if cfg.teacher.top_logprobs else None
    done_a = {r["id"] for r in out.existing()}
    answer_jobs = [(q, k) for q in questions for k in range(gcfg.answers_per_question) if f"{q['id']}/s{k}" not in done_a]
    print(f"[generate] {gcfg.answers_per_question} answers per question; {len(answer_jobs)} to go")

    def make_answer(item):
        q, k = item
        c = teacher.chat(A_SYSTEM, A_PROMPT.format(chunk=q["text"], question=q["question"]), top_logprobs=cfg.teacher.top_logprobs)
        meta = {k_: q[k_] for k_ in ("doc_id", "chunk_id", "persona", "task")}
        row = {"id": f"{q['id']}/s{k}", **prose_row(meta, q["question"], c.content), "sample": k, "teacher": teacher.model}
        if lp_out and c.logprobs:
            lp_out.append({"id": row["id"], **c.logprobs})  # written first: a row never lacks its logprobs
        out.append(row)

    teacher.map(make_answer, answer_jobs)
    print(f"[generate] {len(out.existing())} rows -> {out.path}")
    return out.path
