"""Agent smoke tests on a real Slurm cluster: the same 10 tasks, easy to hard, for every model behind an OpenAI-compatible
endpoint, through `hpc-assistant --agent` (distillkit.assistant.Assistant + distillkit.slurm_live.SlurmLive, actions
auto-confirmed). Fixture jobs (a failing job, a job with a typo, a long sleep, a finished job) are submitted first.
Each task gets a fresh conversation; every tool call and result is recorded, then checked automatically once the
jobs the agent submitted have finished.

usage: smoke.py OUT_DIR ACCOUNT CPU_PARTITION INDEX_DIR [NAME=ENDPOINT[=SERVED_MODEL] ...]
Without models: only submit the fixture jobs (OUT_DIR/fixtures.json); with models: reuse them, so several
processes (one per model and GPU) can run side by side within one 30-minute debug job.
"""
import json
import os
import re
import sys
import time
from io import StringIO
from pathlib import Path

from distillkit.assistant import Assistant
from distillkit.slurm_live import SlurmLive

out_dir, ACCOUNT, PART, INDEX = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
models = [a.split("=", 2) for a in sys.argv[5:]]
out_dir.mkdir(parents=True, exist_ok=True)
JOBS = str(out_dir / "jobs")
HEAD = f"#!/bin/bash\n#SBATCH -A {ACCOUNT}\n#SBATCH -p {PART}\n#SBATCH -c 1\n"
live = SlurmLive(workdir=JOBS, confirm=lambda n, a: True)


def submit(name, body, t="00:02:00"):
    r = live.submit_job(f"{HEAD}#SBATCH -t {t}\n#SBATCH -J {name}\n{body}")
    assert r.get("submitted"), r
    return r["job_id"]


def wait(ids, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        states = {i: live.job_status(i).get("state") for i in ids}
        if all(s not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", None) for s in states.values()):
            return states
        time.sleep(15)
    return {i: live.job_status(i).get("state") for i in ids}


def fixtures():
    fail = submit("smoke-fail", "python3 -c 'import numpy_does_not_exist'\n")
    typo = submit("smoke-typo", "ech \"result: 42\"\n")
    done = submit("smoke-done", "python3 -c 'x=[0]*30_000_000; print(sum(x))'\n")
    wait([fail, typo, done])
    return {"fail": fail, "typo": typo, "done": done}


def tasks(fx, sleeper):
    return [
        ("queue", "List my queued and running jobs.", None),
        ("partition", "What is the maximum wall time and how many GPUs per node does the boost_usr_prod partition have?", None),
        ("gpus", "How many GPUs are free on boost_usr_prod right now?", None),
        ("submit", f"Submit a job on partition {PART} with account {ACCOUNT}: 1 CPU, 2 minutes, named smoke-hello, "
                   "that prints the hostname. Then tell me the job id.", None),
        ("validate", f"Check this script without submitting it:\n```bash\n{HEAD}#SBATCH -t 00:02:00\nsrun hostname\n```", None),
        ("debug", f"My job {fx['fail']} failed. Find out why and tell me how to fix it.", None),
        ("retrigger", f"Job {fx['typo']} failed because of a typo in its command. Fix the script and resubmit it.", None),
        ("cancel", f"Cancel my job {sleeper}.", None),
        ("array", f"Submit a job array on partition {PART} with account {ACCOUNT}: 4 tasks, at most 2 running at a time, "
                  "1 CPU and 2 minutes each, each task prints its array task id.", None),
        ("accounting", f"How much memory and CPU did job {fx['done']} use, and did it finish successfully?", None),
    ]


class Recorder(SlurmLive):
    def __init__(self):
        super().__init__(workdir=JOBS, confirm=lambda n, a: True)
        self.log = []

    def execute(self, name, arguments):
        try:
            res = super().execute(name, arguments)
        except Exception as e:  # recorded, then re-raised for the assistant to report as a tool error
            self.log.append({"tool": name, "arguments": arguments, "error": str(e)})
            raise
        self.log.append({"tool": name, "arguments": arguments, "result": res})
        return res


def run_model(name, endpoint, served, fx):
    sleeper = submit(f"smoke-sleep-{name}", "sleep 600\n", t="00:15:00")
    rows = []
    for tid, question, _ in tasks(fx, sleeper):
        rec = Recorder()
        bot = Assistant(INDEX, model=served or name, openai=endpoint, slurm=rec)
        buf, t0 = StringIO(), time.time()
        try:
            answer, _ = bot.ask(question, out=buf)
            err = None
        except Exception as e:  # a crash is a failed task, not a failed run
            answer, err = buf.getvalue(), repr(e)
        rows.append({"model": name, "task": tid, "question": question, "answer": answer, "tools": rec.log,
                     "error": err, "seconds": round(time.time() - t0, 1)})
        with open(out_dir / f"{name}.partial.jsonl", "a") as f:  # survives the job's time limit
            f.write(json.dumps(rows[-1]) + "\n")
        print(name, tid, [c["tool"] for c in rec.log], err or "", flush=True)
    if live.job_status(sleeper).get("state") in ("PENDING", "RUNNING"):
        live.cancel_job(sleeper)
    return rows, sleeper


def submitted_ids(row):
    return [c["result"]["job_id"] for c in row["tools"]
            if c["tool"] == "submit_job" and isinstance(c.get("result"), dict) and c["result"].get("submitted")]


def args_of(call):
    a = call["arguments"]
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except json.JSONDecodeError:  # the model sent invalid JSON: the tool rejected it too
        return {}


def called(row, tool, **want):
    return any(c["tool"] == tool and "result" in c and "error" not in c["result"] and
               all(args_of(c).get(k) == v for k, v in want.items()) for c in row["tools"])


def check(row, fx, sleeper):
    t, a = row["task"], (row["answer"] or "").lower()
    ids = submitted_ids(row)
    if t == "queue":
        return called(row, "list_queue")
    if t == "partition":
        return called(row, "partition_info") and ("24" in a or "1-00:00:00" in a or "one day" in a) and "4" in a
    if t == "gpus":
        return called(row, "gpu_availability")
    if t == "submit":
        if len(ids) != 1:
            return False
        st = wait(ids)[ids[0]]
        log = live.read_job_log(ids[0], "stdout", 5).get("lines", [])
        return st == "COMPLETED" and any(re.search(r"\w+\.\w+|lrdn|login", l) for l in log) and ids[0] in a
    if t == "validate":
        return called(row, "submit_job", test_only=True) and not ids
    if t == "debug":
        return (called(row, "read_job_log") or called(row, "job_accounting")) and "numpy_does_not_exist" in a
    if t == "retrigger":
        if not called(row, "job_script") or len(ids) != 1:
            return False
        st = wait(ids)[ids[0]]
        log = live.read_job_log(ids[0], "stdout", 5).get("lines", [])
        return st == "COMPLETED" and any("result: 42" in l for l in log)
    if t == "cancel":
        return called(row, "cancel_job", job_id=sleeper) and live.job_status(sleeper).get("state", "").startswith("CANCELLED")
    if t == "array":
        if len(ids) != 1:
            return False
        script = next((args_of(c).get("script", "") for c in row["tools"]
                       if c["tool"] == "submit_job" and isinstance(c.get("result"), dict) and c["result"].get("submitted")), "")
        ok_spec = re.search(r"--array[= ](0-3|1-4)%2\b", script) is not None
        wait([ids[0]])
        return ok_spec and "slurm_array_task_id" in script.lower()
    if t == "accounting":
        return called(row, "job_accounting", job_id=fx["done"]) and "complet" in a
    return False


FX = out_dir / "fixtures.json"
if not models:
    FX.write_text(json.dumps(fixtures()))
    print("fixtures", FX.read_text(), flush=True)
    sys.exit(0)
fx = json.loads(FX.read_text())
results = {}
for name, endpoint, *served in models:
    partial = out_dir / f"{name}.partial.jsonl"
    if os.environ.get("RESCORE") and partial.exists():  # score recorded rows again (checker fixes), no model calls
        rows = [json.loads(l) for l in partial.read_text().splitlines()]
        sleeper = re.search(r"job (\d+)", next(r["question"] for r in rows if r["task"] == "cancel")).group(1)
    else:
        rows, sleeper = run_model(name, endpoint, served[0] if served else None, fx)
    for r in rows:
        r["pass"] = bool(check(r, fx, sleeper))
    results[name] = rows
    (out_dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(name, "passed", sum(r["pass"] for r in rows), "/", len(rows), flush=True)
for n, rows in results.items():
    summary = {"passed": sum(r["pass"] for r in rows), "tasks": {r["task"]: r["pass"] for r in rows},
               "seconds": sum(r["seconds"] for r in rows), "fixtures": fx}
    (out_dir / f"SUMMARY_{n}.json").write_text(json.dumps(summary, indent=1))
    print(n, json.dumps(summary))
