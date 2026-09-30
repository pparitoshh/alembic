"""Stage 1: document-grounded question generation + multiple grounded answers per question."""

import itertools
import random
from pathlib import Path

from .io import write_jsonl
from .seeds import load_chunks
from .teacher import Teacher

Q_SYSTEM = "You write realistic questions that users of an HPC cluster ask. Output only the question text, nothing else."

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
Write it in the persona's own words. Output only the question."""

A_SYSTEM = """You are an expert HPC assistant. Answer concisely and correctly.
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
        q = teacher.chat(Q_SYSTEM, Q_PROMPT.format(chunk=j["text"], persona=j["persona"], task=j["task"]))
        return {**j, "question": q.strip().strip('"')}

    questions = teacher.map(make_question, jobs)

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
