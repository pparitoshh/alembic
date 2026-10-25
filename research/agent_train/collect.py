"""Agent training data from real Slurm: a teacher model (Qwen3-32B, 10/10 on research/agent_smoke) solves randomized
tasks through the same harness as the student (distillkit.assistant.Assistant + distillkit.slurm_live.SlurmLive, real
squeue/sacct/sbatch/scancel/sinfo/scontrol), every trajectory is checked against the real outcome, and only verified
trajectories are kept, with the full message history (system, user turn with retrieved excerpts, assistant tool calls,
real tool results, final answer).

The 10 tasks of research/agent_smoke stay held out: the task families here cover the same skills with other wording,
other failures, job names, partitions, sizes and multi-step flows, plus families the smoke test does not have
(dependencies, cancel by name, job status, conditional fix-and-resubmit).

usage: collect.py OUT_DIR ACCOUNT CPU_PARTITION INDEX_DIR ENDPOINT SERVED_MODEL N_TASKS SEED [CONCURRENCY]
"""
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

from distillkit.assistant import Assistant
from distillkit.slurm_live import SlurmLive

out_dir, ACCOUNT, PART, INDEX, ENDPOINT, SERVED = Path(sys.argv[1]), *sys.argv[2:7]
N, SEED = int(sys.argv[7]), int(sys.argv[8])
CONC = int(sys.argv[9]) if len(sys.argv) > 9 else 12
out_dir.mkdir(parents=True, exist_ok=True)
JOBS = str(out_dir / f"jobs_{SEED}")
HEAD = f"#!/bin/bash\n#SBATCH -A {ACCOUNT}\n#SBATCH -p {PART}\n#SBATCH -c 1\n#SBATCH --mem=1G\n"
live = SlurmLive(workdir=JOBS, confirm=lambda n, a: True)
rng = random.Random(SEED)
lock = threading.Lock()
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kepler",
         "lumen", "marble", "nimbus", "onyx", "prism", "quartz", "raven", "sierra", "tundra", "umber", "vortex"]


SLOTS = out_dir / "slots"  # lrd_all_serial allows 2 running / 10 submitted jobs per user (array tasks count singly): at most 4 submitting tasks
SLOTS.mkdir(exist_ok=True)


class slot:
    """Cluster-wide token (atomic mkdir on the shared file system) for tasks that submit real jobs."""

    def __enter__(self):
        while True:
            for k in range(4):
                d = SLOTS / f"s{k}"
                try:
                    d.mkdir()
                    self.d = d
                    return self
                except FileExistsError:
                    if time.time() - d.stat().st_mtime > 1200:  # stale token of a killed collector
                        d.rmdir()
            time.sleep(5)

    def __exit__(self, *a):
        self.d.rmdir()


def name():
    return f"{rng.choice(WORDS)}-{rng.randint(10, 99)}"


def submit(jname, body, t="00:02:00"):
    r = live.submit_job(f"{HEAD}#SBATCH -t {t}\n#SBATCH -J {jname}\n{body}")
    assert r.get("submitted"), r
    return r["job_id"]


def wait(ids, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = {i: live.job_status(i).get("state") for i in ids}
        if all(s not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", None) for s in st.values()):
            return st
        time.sleep(10)
    return {i: live.job_status(i).get("state") for i in ids}


def log_lines(jid):
    return live.read_job_log(jid, "stdout", 50).get("lines", [])


# ---- fixtures: finished jobs with known failures, finished successes, long sleepers (cancel targets) ---------------
FAILS = [  # (body, token the diagnosis must name)
    ("python3 -c 'import {m}'\n", "{m}"),
    ("{c} --version\n", "{c}"),
    ("cat input_{w}.csv\n", "input_{w}.csv"),
    ("echo starting; exit {e}\n", "{e}"),
    ("python3 -c 'print(1/0)'\n", "ZeroDivisionError"),
]
FIXABLE = [  # (broken body, expected output after the fix)
    ("ech \"done {w}\"\n", "done {w}"),
    ("pyhton3 -c 'print(\"{w} ok\")'\n", "{w} ok"),
    ("prinf '%s\\n' \"{w} ready\"\n", "{w} ready"),
    ("echo \"{w} start\" && sleeep 1 && echo \"{w} end\"\n", "{w} end"),
]


def make_fixtures(k):
    fx = {"fail": [], "fix": [], "done": []}
    for _ in range(k):
        body, tok = rng.choice(FAILS)
        p = {"m": f"mod_{rng.choice(WORDS)}", "c": f"tool{rng.randint(100, 999)}x", "w": rng.choice(WORDS),
             "e": rng.choice([2, 3, 4, 7, 42])}
        fx["fail"].append((submit(name(), body.format(**p)), tok.format(**p)))
        body, want = rng.choice(FIXABLE)
        w = rng.choice(WORDS)
        fx["fix"].append((submit(name(), body.format(w=w)), want.format(w=w)))
        n = rng.choice([1, 5, 20]) * 1_000_000
        fx["done"].append(submit(name(), f"python3 -c 'x=[0]*{n}; print(len(x))'\n"))
    wait([j for j, _ in fx["fail"]] + [j for j, _ in fx["fix"]] + fx["done"])
    return fx


def new_sleeper():
    """A cancel target that stays PENDING (start time in 2 hours): it never takes one of the 2 running slots."""
    jn = name()
    r = live.submit_job(f"{HEAD}#SBATCH -t 00:05:00\n#SBATCH -J {jn}\n#SBATCH --begin=now+2hours\nsleep 60\n")
    assert r.get("submitted"), r
    return r["job_id"], jn


# ---- task families: (question, checker(row) -> bool) --------------------------------------------------------------
def ids_submitted(row):
    return [c["result"]["job_id"] for c in row["tools"] if c["tool"] == "submit_job"
            and isinstance(c.get("result"), dict) and c["result"].get("submitted")]


def args_of(c):
    a = c["arguments"]
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a or "{}")
    except json.JSONDecodeError:
        return {}


def called(row, tool, **want):
    return any(c["tool"] == tool and isinstance(c.get("result"), dict) and "error" not in c["result"]
               and all(str(args_of(c).get(k)) == str(v) for k, v in want.items()) for c in row["tools"])


FAMS = ["queue", "partition", "gpus", "submit", "validate", "debug", "retrigger", "cancel_id",
        "cancel_name", "array", "accounting", "dependency", "status", "fix_if_failed"]


def task(fx, fam):
    if fam == "queue":
        q = rng.choice(["Which of my jobs are in the queue right now?", "Do I have anything pending or running?",
                        "Show me my running jobs.", "Are any of my jobs still waiting to start?"])
        return fam, q, lambda r: called(r, "list_queue")
    if fam == "partition":
        part = rng.choice(["boost_usr_prod", "lrd_all_serial", "dcgp_usr_prod"])
        what = rng.choice(["maximum wall time", "number of CPUs per node", "memory per node", "limits"])
        return fam, f"What is the {what} of the {part} partition?", lambda r, p=part: called(r, "partition_info", partition=p)
    if fam == "gpus":
        q = rng.choice(["Are there free GPUs on boost_usr_prod at the moment?", "How many A100s are idle on boost_usr_prod?",
                        "Check GPU availability on the booster partition (boost_usr_prod)."])
        return fam, q, lambda r: called(r, "gpu_availability")
    if fam == "submit":
        jn, w, t = name(), rng.choice(WORDS), rng.choice([1, 2, 3])
        cmd = rng.choice([f"echo {w}", f"python3 -c 'print(\"{w}\")'", f"printf '%s\\n' {w}"])
        q = (f"Please submit a batch job named {jn} to partition {PART}, account {ACCOUNT}, with 1 CPU, 1 GB of memory "
             f"and a {t}-minute limit, that runs: {cmd}. Give me the job id.")

        def chk(r, w=w):
            ids = ids_submitted(r)
            return len(ids) == 1 and wait(ids)[ids[0]] == "COMPLETED" and any(w in l for l in log_lines(ids[0])) \
                and ids[0] in (r["answer"] or "")
        return fam, q, chk
    if fam == "validate":
        gpu = rng.random() < 0.5
        script = (f"#!/bin/bash\n#SBATCH -A {ACCOUNT}\n#SBATCH -p boost_usr_prod\n#SBATCH --gres=gpu:{rng.choice([1, 2, 4])}\n"
                  f"#SBATCH -t 00:10:00\nsrun nvidia-smi\n") if gpu else f"{HEAD}#SBATCH -t 00:05:00\nsrun hostname\n"
        q = rng.choice(["Is this script valid? Only test it, do not submit it.",
                        "Before I submit this, can you check that Slurm would accept it (dry run only)?"])
        return fam, f"{q}\n```bash\n{script}```", lambda r: called(r, "submit_job", test_only=True) and not ids_submitted(r)
    if fam == "debug":
        jid, tok = rng.choice(fx["fail"])
        q = rng.choice([f"Job {jid} did not work. What went wrong?", f"Why did my job {jid} fail?",
                        f"Can you diagnose job {jid} for me?"])
        return fam, q, lambda r, j=jid, t=tok: (called(r, "read_job_log") or called(r, "job_accounting")) and \
            t.lower() in (r["answer"] or "").lower()
    if fam == "retrigger":
        jid, want = rng.choice(fx["fix"])
        q = rng.choice([f"Job {jid} has a typo in its command. Correct the script and submit it again.",
                        f"Please fix whatever is wrong in job {jid}'s script and rerun it."])

        def chk(r, want=want):
            ids = ids_submitted(r)
            return called(r, "job_script") and len(ids) == 1 and wait(ids)[ids[0]] == "COMPLETED" and \
                any(want in l for l in log_lines(ids[0]))
        return fam, q, chk
    if fam in ("cancel_id", "cancel_name"):
        jid, jn = new_sleeper()
        q = f"Cancel job {jid}." if fam == "cancel_id" else f"Please cancel my job called {jn}."

        def chk(r, jid=jid):
            others = [args_of(c).get("job_id") for c in r["tools"] if c["tool"] == "cancel_job"]
            return called(r, "cancel_job", job_id=jid) and set(others) == {jid} and \
                live.job_status(jid).get("state", "").startswith("CANCELLED")
        return fam, q, chk
    if fam == "array":
        n, k = rng.choice([2, 3, 4]), rng.choice([1, 2])
        q = (f"Submit a job array to {PART} (account {ACCOUNT}) with {n} tasks, no more than {k} running at once, "
             f"1 CPU, 1 GB and 2 minutes per task; each task should print its own task index.")

        def chk(r, n=n, k=k):
            ids = ids_submitted(r)
            sc = next((args_of(c).get("script", "") for c in r["tools"] if c["tool"] == "submit_job"
                       and isinstance(c.get("result"), dict) and c["result"].get("submitted")), "")
            return len(ids) == 1 and re.search(rf"--array[= ](0-{n - 1}|1-{n})%{k}\b", sc) is not None and \
                "SLURM_ARRAY_TASK_ID" in sc
        return fam, q, chk
    if fam == "accounting":
        jid = rng.choice(fx["done"])
        q = rng.choice([f"How long did job {jid} run and how much memory did it use?",
                        f"What was the final state and CPU efficiency of job {jid}?"])
        return fam, q, lambda r, j=jid: called(r, "job_accounting", job_id=j) and "complet" in (r["answer"] or "").lower()
    if fam == "dependency":
        a, b = rng.sample(WORDS, 2)
        q = (f"Submit two jobs to {PART} (account {ACCOUNT}, 1 CPU, 1 GB, 2 minutes each): the first prints {a}; the "
             f"second prints {b} and must start only if the first one finishes successfully.")

        def chk(r):
            subs = [(c["result"]["job_id"], args_of(c).get("script", "")) for c in r["tools"] if c["tool"] == "submit_job"
                    and isinstance(c.get("result"), dict) and c["result"].get("submitted")]
            return len(subs) == 2 and re.search(rf"--dependency[= ]afterok:{subs[0][0]}\b", subs[1][1]) is not None
        return fam, q, chk
    if fam == "status":
        jid = rng.choice(fx["done"]) if rng.random() < 0.6 else new_sleeper()[0]
        return fam, rng.choice([f"Is job {jid} still running?", f"What is the state of job {jid}?"]), \
            lambda r, j=jid: called(r, "job_status", job_id=j)
    # fix_if_failed: conditional multi-step
    jid, want = rng.choice(fx["fix"])
    q = f"Check job {jid}. If it failed, find out why, fix the script and resubmit it; otherwise just tell me it is fine."

    def chk(r, jid=jid, want=want):
        ids = ids_submitted(r)
        return (called(r, "job_status", job_id=jid) or called(r, "job_accounting", job_id=jid)) and \
            called(r, "job_script") and len(ids) == 1 and wait(ids)[ids[0]] == "COMPLETED" and \
            any(want in l for l in log_lines(ids[0]))
    return "fix_if_failed", q, chk


class Recorder(SlurmLive):
    def __init__(self):
        super().__init__(workdir=JOBS, confirm=lambda n, a: True)
        self.log = []

    def execute(self, name, arguments):
        try:
            res = super().execute(name, arguments)
        except Exception as e:
            self.log.append({"tool": name, "arguments": arguments, "error": str(e)})
            raise
        self.log.append({"tool": name, "arguments": arguments, "result": res})
        return res


# Prompt distillation: the teacher (thinking off) tends to answer from job_status alone; this guidance is given to the
# teacher only, so the student learns the behaviour itself from the verified trajectories.
TEACHER_HINT = ("\n\nWhen you have tools, use them to get facts before you answer: to explain why a job failed, read "
                "its log (read_job_log; stdout and stderr may be one file) and its accounting; to fix and resubmit a job, "
                "get its script with job_script first, then submit the corrected script; to cancel a job given by name, "
                "list the queue first to find its id; for a job array use --array=FIRST-LAST%MAX and "
                "$SLURM_ARRAY_TASK_ID; after submitting, report the job id.")
ONLY = [f for f in os.environ.get("FAMILIES", "").split(",") if f]  # e.g. FAMILIES=debug,fix_if_failed
SUBMITS = {"submit", "retrigger", "cancel_id", "cancel_name", "array", "dependency", "fix_if_failed", "status"}


def run_one(i, fx):
    with lock:
        fam = rng.choice(ONLY or FAMS)
    if fam in SUBMITS:
        with slot():
            return attempt(i, fx, fam)
    return attempt(i, fx, fam)


def attempt(i, fx, fam):
    with lock:
        fam, q, chk = task(fx, fam)
    rec = Recorder()
    bot = Assistant(INDEX, model=SERVED, openai=ENDPOINT, slurm=rec)
    # THINK=1: the teacher reasons first (Qwen3-32B with thinking passed the debug task; without it, it answers from
    # job_status alone); build_agent_data.py removes the <think> text, so the student learns the actions
    bot.extra_body = {"chat_template_kwargs": {"enable_thinking": os.environ.get("THINK") == "1"}}
    system = bot.history[0]["content"]
    bot.history[0]["content"] = system + TEACHER_HINT  # teacher only: training uses the plain system prompt
    try:
        answer, _ = bot.ask(q, out=StringIO())
        err = None
    except Exception as e:
        answer, err = "", repr(e)
    bot.history[0]["content"] = system  # the stored conversation carries the student's system prompt
    row = {"id": f"agent-{SEED}-{i}", "family": fam, "question": q, "answer": answer, "tools": rec.log, "error": err,
           "messages": bot.history}
    try:
        row["pass"] = bool(not err and chk(row))
    except Exception as e:
        row["pass"], row["check_error"] = False, repr(e)
    with lock, open(out_dir / f"traj_{SEED}.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return fam, row["pass"]


FXF, FXL = out_dir / "fixtures.json", out_dir / "fixtures.lock"
try:
    FXL.mkdir()  # the first collector of this run builds the shared fixtures
    FXF.write_text(json.dumps(make_fixtures(3)))
except FileExistsError:
    while not FXF.exists():
        time.sleep(10)
fx = json.loads(FXF.read_text())
fx = {"fail": [tuple(x) for x in fx["fail"]], "fix": [tuple(x) for x in fx["fix"]], "done": fx["done"]}
stats = {}
with ThreadPoolExecutor(CONC) as ex:
    for fam, ok in ex.map(lambda i: run_one(i, fx), range(N)):
        s = stats.setdefault(fam, [0, 0])
        s[0] += ok
        s[1] += 1
print(json.dumps(stats), flush=True)
for j in live.list_queue().get("jobs", []):  # leftover sleepers from cancel tasks the teacher failed
    if j["name"].split("-")[0] in WORDS and j["state"] in ("PENDING", "RUNNING"):
        live.cancel_job(j["job_id"])
