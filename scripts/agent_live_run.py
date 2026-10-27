"""Live agent task runner: solves data/agent_live/tasks.jsonl prompts in-process with the assistant's agent mode.

usage: agent_live_run.py [--ids ID,...] [--limit N] [--index DIR]
                       [--running-job JID] [--pending-job JID] [--finished-job JID]

Each task gets a fresh SlurmLive + Assistant. The model is the Ollama student
(hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest) and the Slurm commands run on
leonardo over the already-open SSH ControlMaster (/tmp/cl-%C). `confirm` always
declines, so cluster-changing actions are never executed: this run measures tool
selection, not changes (no --yes logic exists here on purpose).

Writes one JSON line per task to data/agent_live/runs/<timestamp>.jsonl
(id, prompt, calls with results truncated to 2000 characters, answer, seconds).
No side effects at import: everything happens in `main()` under `__main__`.
"""
import argparse
import concurrent.futures
import json
import shlex
import subprocess
import time
from io import StringIO
from pathlib import Path

from distillkit.assistant import DEFAULT_INDEX, Assistant
from distillkit.slurm_live import SlurmLive

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/agent_live/tasks.jsonl"
RUNS = ROOT / "data/agent_live/runs"
SSH = ["ssh", "-o", "ControlPath=/tmp/cl-%C", "-o", "ControlMaster=no", "-o", "BatchMode=yes", "leonardo"]
MODEL = "hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest"

class Recorder(SlurmLive):
    """SlurmLive with the run() options for the shared ControlMaster, recording every tool call."""

    def __init__(self):
        super().__init__(confirm=lambda name, args: False)  # actions are always declined here
        self.log = []

    def run(self, argv, stdin=None):
        p = subprocess.run(SSH + [shlex.join(argv)], input=stdin, capture_output=True, text=True,
                           timeout=60 + self.timeout)
        return p.returncode, p.stdout, p.stderr

    def execute(self, name, arguments):
        try:
            res = super().execute(name, arguments)
        except Exception as e:
            self.log.append({"name": name, "args": arguments, "result": f"error: {e}"[:2000]})
            raise
        self.log.append({"name": name, "args": arguments, "result": json.dumps(res)[:2000]})
        return res


def fill(prompt, placeholders):
    for tag, value in placeholders.items():
        prompt = prompt.replace(tag, value)
    return prompt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", help="comma-separated task ids to run (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many tasks")
    ap.add_argument("--index", default=DEFAULT_INDEX, help="assistant index directory")
    ap.add_argument("--model", default=MODEL, help="Ollama model name for the agent")
    ap.add_argument("--running-job", default="<RUNNING_JOB>")
    ap.add_argument("--pending-job", default="<PENDING_JOB>")
    ap.add_argument("--finished-job", default="<FINISHED_JOB>")
    ap.add_argument("--task-timeout", type=float, default=180.0,
                    help="wall-clock seconds per task; the task is recorded with timeout: true and the run continues")
    args = ap.parse_args(argv)
    placeholders = {"<RUNNING_JOB>": args.running_job, "<PENDING_JOB>": args.pending_job,
                    "<FINISHED_JOB>": args.finished_job}
    tasks = [json.loads(l) for l in TASKS.read_text().splitlines() if l.strip()]
    if args.ids:
        want = set(args.ids.split(","))
        tasks = [t for t in tasks if t["id"] in want]
    if args.limit:
        tasks = tasks[:args.limit]
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = RUNS / f"{stamp}.jsonl"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for t in tasks:
            prompt, t0 = fill(t["prompt"], placeholders), time.time()
            rec = Recorder()
            bot = Assistant(args.index, model=args.model, slurm=rec)
            timed_out = False
            with StringIO() as buf:
                fut = pool.submit(bot.ask, prompt, buf)
                try:
                    answer, _ = fut.result(timeout=args.task_timeout)
                except concurrent.futures.TimeoutError:
                    timed_out = True
                    answer = ""
            row = {"id": t["id"], "prompt": prompt, "calls": rec.log, "answer": answer,
                   "seconds": round(time.time() - t0, 1)}
            if timed_out:
                row["timeout"] = True
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
