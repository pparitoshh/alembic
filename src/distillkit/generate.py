"""Stage 1: document-grounded question generation + multiple grounded answers per question."""

import itertools
import random
from pathlib import Path

from .io import write_jsonl
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


def run(cfg: dict) -> Path:
    gcfg = cfg["generate"]
    teacher = Teacher(cfg)
    train_chunks, _ = load_chunks(cfg)
    rng = random.Random(0)
    grid = list(itertools.product(gcfg["personas"], gcfg["task_types"]))

    jobs = []
    for chunk in train_chunks:
        for persona, task in rng.sample(grid, min(gcfg["questions_per_chunk"], len(grid))):
            jobs.append({**chunk, "persona": persona, "task": task})

    print(f"[generate] {len(train_chunks)} train chunks -> {len(jobs)} questions")

    def make_question(j):
        q, _ = teacher.chat_json(Q_SYSTEM, Q_PROMPT.format(chunk=j["text"], persona=j["persona"], task=j["task"]), GeneratedQuestion)
        return {**j, "question": q.question.strip()} if q else None

    questions = [q for q in teacher.map(make_question, jobs) if q]
    if len(questions) < len(jobs):
        print(f"[generate] dropped {len(jobs) - len(questions)} questions with invalid JSON")

    answer_jobs = [(q, k) for q in questions for k in range(gcfg["answers_per_question"])]
    print(f"[generate] {len(answer_jobs)} answers ({gcfg['answers_per_question']} per question)")

    def make_answer(item):
        q, k = item
        a = teacher.chat(A_SYSTEM, A_PROMPT.format(chunk=q["text"], question=q["question"]))
        return {
            "doc_id": q["doc_id"],
            "chunk_id": q["chunk_id"],
            "persona": q["persona"],
            "task": q["task"],
            "question": q["question"],
            "answer": a,
            "sample": k,
            "teacher": teacher.model,
        }

    rows = teacher.map(make_answer, answer_jobs)
    out = Path(cfg["run_dir"]) / "generated.jsonl"
    write_jsonl(out, rows)
    print(f"[generate] wrote {len(rows)} rows -> {out}")
    return out
