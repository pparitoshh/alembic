"""Resumable GPU allocation wrapper for the private comparison/export controllers."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from evidence import EvidenceError, read_json
from supervisor import ACTIVE, RETRYABLE, SchedulerError, Slurm, atomic, lock, now


def main(config_path, args):
    config = read_json(config_path)
    directory = Path(config["root"]) / "ctl/autopilot"
    for arg in args:
        if arg.startswith("--nodes=") and not 1 <= int(arg.split("=", 1)[1]) <= 8:
            raise EvidenceError("A GPU job would exceed the existing eight-node/32-GPU limit")
    context = {k: os.environ.get(k) for k in ("RUN", "STAGE", "OUT", "ARM", "ANSWERS", "GGUF_DIR", "QUANTS", "BASE_MODEL")}
    key = hashlib.sha256(json.dumps([args, context], sort_keys=True).encode()).hexdigest()[:20]
    state_path = directory / "gpu" / (key + ".json")
    scheduler = Slurm(config, directory)
    # One owner for the shared priority marker. No controller can delete another controller's reservation.
    while True:
        with lock(directory / "gpu_priority.lock") as acquired:
            if not acquired:
                time.sleep(10)
                continue
            state = read_json(state_path) if state_path.exists() else {"attempt": 0}
            if state.get("job_id"):
                status, _ = scheduler.status(state["job_id"])
                if status not in ACTIVE and status != "UNKNOWN":
                    state.pop("job_id")
            marker = Path(config["root"]) / "ctl/PRIORITY"
            owner = "autopilot:" + key
            marker.write_text(owner + "\n")
            try:
                while not state.get("job_id"):
                    if (directory / "STOP").exists():
                        return 1
                    if (directory / "PAUSE").exists():
                        time.sleep(30)
                        continue
                    try:
                        attempt = state["attempt"] + 1
                        job = scheduler.submit(f"gpu-{key}-{attempt}", args[-1], gpu_args=args)
                        state.update(job_id=job, attempt=attempt, submitted_at=now())
                        atomic(state_path, state)
                    except SchedulerError as exc:
                        print(str(exc), flush=True)
                        time.sleep(30)
                while True:
                    try:
                        status, exit_code = scheduler.status(state["job_id"])
                    except SchedulerError:
                        time.sleep(30)
                        continue
                    if status in ACTIVE or status == "UNKNOWN":
                        time.sleep(30)
                        continue
                    state.update(state=status, exit_code=exit_code, finished_at=now())
                    atomic(state_path, state)
                    print(json.dumps(state), flush=True)
                    # Time-limited jobs leave resumable unit files. The caller rechecks coverage before retrying.
                    return 0 if (status == "COMPLETED" and exit_code == "0:0") or status in RETRYABLE else 1
            finally:
                if marker.exists() and marker.read_text().strip() == owner:
                    marker.unlink()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2:]))
