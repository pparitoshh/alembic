"""Stage 1: document-grounded questions + grounded answers or tool-calling traces.

Each question gets a mode: `prose` (plain answer), or, for `generate.tool_fraction` of them,
a tool mode: `call` (needs live cluster info, so the teacher calls a tool against the mock
cluster), `ask` (needs a tool, but a required detail is missing, so the teacher asks back),
or `none` (tools are offered but not needed). The student learns both how and when to call.

Resumable: questions and answers are appended to the run dir as they finish, and a rerun only
does the missing ones. On a shared cluster node, a job killed at its wall time loses nothing.
"""

import itertools
import json
import random
from pathlib import Path

from .config import Config
from .io import JsonlAppender
from .records import prose_row
from .schemas import GeneratedQuestion
from .seeds import load_chunks
from .teacher import Teacher
from .tools import PARTITIONS, SCHEMAS, ToolError, execute, parse_arguments

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
{mode_rule}
The question must not mention "the documentation".
Write it in the persona's own words. Reply only with JSON: {{"question": "<the question>"}}."""

MODE_RULES = {
    "prose": "The question must be answerable from the documentation above.",
    "none": "The question must be answerable from the documentation above, without looking at the user's jobs or the cluster's current state.",
    "call": """The user has an assistant with these tools for their cluster:
{tools}
The question must need live information or an action that one of these tools provides (a job's state, logs or
resource usage, the queue, free GPUs, partition limits, submitting or cancelling a job), related to the topic of
the documentation above. Include every detail the tool needs. If you mention a job, use job id {job_id}: on this
cluster it is {state} in partition {partition}, so keep the question consistent with that. Partitions here are
{partitions}.""",
    "ask": """The user has an assistant with these tools for their cluster:
{tools}
The question must need one of these tools, but leave out a detail the tool requires, e.g. talk about "my job"
without giving its id, so the assistant has to ask for it first. Relate it to the topic of the documentation above.""",
}

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

TOOL_RULES = """
You can call tools that query and act on the user's Slurm cluster:
- Call a tool only when the answer needs live cluster information (a job's state, logs, usage, the queue, free GPUs,
  partition limits) or the user asks you to act (submit, cancel). Answer general questions directly, without tools.
- Never invent job ids or other arguments. If a required detail is missing, ask the user for it in one short
  sentence and do not call a tool.
- Before a call, say in one short sentence what you will check. After the result, answer briefly from it and
  give the fix if something failed.

Reference material (ground truth for general Slurm facts; never mention it):
<doc>
{chunk}
</doc>"""


def _gold(cfg: Config, name: str) -> str:
    """Gold transcript rendered as a labelled example for the system prompt. As earlier chat turns it
    leaked: the teacher took the example's job id for the user's own job."""
    if not cfg.generate.gold_dir:
        return ""
    path = Path(cfg.generate.gold_dir) / f"{name}.json"
    if not path.exists():
        return ""
    lines = []
    for m in json.loads(path.read_text())["messages"]:
        if m["role"] == "tool":
            lines.append(f"[tool result] {m['content']}")
            continue
        if m.get("content"):
            lines.append(f"[{m['role']}] {m['content']}")
        for c in m.get("tool_calls") or []:
            lines.append(f"[assistant calls] {c['function']['name']}({json.dumps(c['function']['arguments'])})")
    return (
        "\n\nExample of the expected style, from a different user and job; never reuse its details:\n<example>\n"
        + "\n".join(lines)
        + "\n</example>"
    )


def to_api(messages: list[dict]) -> list[dict]:
    """Record-format transcript -> OpenAI request messages (tool call ids, JSON-string arguments)."""
    out, n, pending = [], 0, []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                n += 1
                f = c["function"]
                args = f["arguments"] if isinstance(f["arguments"], str) else json.dumps(f["arguments"])
                calls.append({"id": f"call_{n}", "type": "function", "function": {"name": f["name"], "arguments": args}})
            pending = [c["id"] for c in calls]  # tool results follow in call order
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0) if pending else f"call_{n}", "content": m["content"]})
        else:
            out.append(m)
    return out


def _record_call(c: dict) -> dict:
    """OpenAI tool call -> record format (no id; arguments as a dict when they parse)."""
    f = c["function"]
    try:
        args = parse_arguments(f["arguments"])
    except ToolError:
        args = f["arguments"]  # kept as is, so verify rejects it
    return {"type": "function", "function": {"name": f["name"], "arguments": args}}


def tool_trace(teacher: Teacher, cfg: Config, chunk: str, question: str, gold: str = "") -> list[dict]:
    """Run the teacher as an agent against the mock cluster; returns the transcript after the system prompt."""
    system = {"role": "system", "content": A_SYSTEM + "\n" + TOOL_RULES.format(chunk=chunk) + gold}
    convo = [{"role": "user", "content": question}]
    for _ in range(cfg.generate.max_tool_rounds + 1):
        c = teacher.complete([system, *to_api(convo)], tools=SCHEMAS)
        if not c.tool_calls:
            convo.append({"role": "assistant", "content": c.content})
            return convo
        calls = [_record_call(tc) for tc in c.tool_calls]
        convo.append({"role": "assistant", "content": c.content, "tool_calls": calls})
        for call in calls:
            try:
                result = execute(call["function"]["name"], call["function"]["arguments"])
            except ToolError as e:
                result = {"error": str(e)}
            convo.append({"role": "tool", "content": json.dumps(result)})
    return convo  # ran out of rounds: ends on a tool result, so verify rejects it (empty final answer)


def question_jobs(cfg: Config) -> list[dict]:
    """Deterministic (chunk, persona, task, mode) sample; `id` is stable across reruns of the same config."""
    gcfg = cfg.generate
    train_chunks, _ = load_chunks(cfg)
    rng = random.Random(0)
    grid = list(itertools.product(range(len(gcfg.personas)), gcfg.task_types))
    modes, weights = zip(*gcfg.tool_mix.items())
    jobs = []
    for chunk in train_chunks:
        for p, task in rng.sample(grid, min(gcfg.questions_per_chunk, len(grid))):
            jid = f"{chunk['chunk_id']}/{task}/p{p}"
            # per-job rng: the prose sample above (and its ids) doesn't change when tool_fraction does
            r = random.Random(f"mode:{jid}")
            mode = r.choices(modes, weights)[0] if r.random() < gcfg.tool_fraction else "prose"
            if mode != "prose":
                jid += f"/{mode}"
            jobs.append({**chunk, "id": jid, "persona": gcfg.personas[p], "task": task, "mode": mode})
    return jobs


def _mode_rule(job: dict) -> str:
    tools = "\n".join(f"- {s['name']}: {s['description']}" for s in SCHEMAS)
    job_id = str(4000000 + random.Random(job["id"]).randrange(1000000))
    mock = execute("job_status", {"job_id": job_id})  # the question's premise must match the mock cluster
    return MODE_RULES[job["mode"]].format(
        tools=tools, job_id=job_id, state=mock["state"], partition=mock["partition"], partitions=", ".join(PARTITIONS)
    )


def run(cfg: Config) -> Path:
    gcfg, run_dir = cfg.generate, cfg.run_dir
    teacher = Teacher(cfg.teacher)
    jobs = question_jobs(cfg)
    gold_prose, gold_trace = _gold(cfg, "prose"), _gold(cfg, "tool_trace")

    q_out = JsonlAppender(run_dir / "questions.jsonl")
    done_q = {q["id"]: q for q in q_out.existing()}
    todo_q = [j for j in jobs if j["id"] not in done_q]
    n_tool = sum(j["mode"] != "prose" for j in jobs)
    print(f"[generate] {len(jobs)} questions planned ({n_tool} tool-mode), {len(done_q)} already done, {len(todo_q)} to go")

    def make_question(j):
        prompt = Q_PROMPT.format(chunk=j["text"], persona=j["persona"], task=j["task"], mode_rule=_mode_rule(j))
        q, _ = teacher.chat_json(Q_SYSTEM, prompt, GeneratedQuestion)
        if q:
            q_out.append({**j, "question": q.question.strip()})

    teacher.map(make_question, todo_q)
    planned = {j["id"] for j in jobs}
    questions = [{"mode": "prose", **q} for q in q_out.existing() if q["id"] in planned]  # old runs have no mode
    if len(questions) < len(jobs):
        print(f"[generate] {len(jobs) - len(questions)} questions had invalid JSON; rerun to retry them")

    out = JsonlAppender(run_dir / "generated.jsonl")
    lp_out = JsonlAppender(run_dir / "teacher_logprobs.jsonl.gz") if cfg.teacher.top_logprobs else None
    done_a = {r["id"] for r in out.existing()}
    answer_jobs = [(q, k) for q in questions for k in range(gcfg.answers_per_question) if f"{q['id']}/s{k}" not in done_a]
    print(f"[generate] {gcfg.answers_per_question} answers per question; {len(answer_jobs)} to go")

    def make_answer(item):
        q, k = item
        meta = {k_: q[k_] for k_ in ("doc_id", "chunk_id", "persona", "task", "mode")}
        head = {"id": f"{q['id']}/s{k}"}
        tail = {"sample": k, "teacher": teacher.model}
        if q["mode"] == "prose":
            msgs = [{"role": "system", "content": A_SYSTEM + gold_prose}, {"role": "user", "content": A_PROMPT.format(chunk=q["text"], question=q["question"])}]
            c = teacher.complete(msgs, top_logprobs=cfg.teacher.top_logprobs)
            row = {**head, **prose_row(meta, q["question"], c.content), **tail}
            if lp_out and c.logprobs:
                lp_out.append({"id": row["id"], **c.logprobs})  # written first: a row never lacks its logprobs
        else:
            # logprobs are not captured for multi-turn traces yet (v2: one entry per assistant turn)
            convo = tool_trace(teacher, cfg, q["text"], q["question"], gold_trace)
            row = {**head, **meta, "question": q["question"], "messages": convo, "tools": SCHEMAS, **tail}
        out.append(row)

    teacher.map(make_answer, answer_jobs)
    print(f"[generate] {len(out.existing())} rows -> {out.path}")
    return out.path
