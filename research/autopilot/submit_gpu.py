"""Return a durable GPU job ID to the resumable training controller."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from evidence import EvidenceError, read_json
from supervisor import ACTIVE, Slurm, atomic, lock, now


def submit(config_path, args):
    config = read_json(config_path)
    directory = Path(config["root"]) / "ctl/autopilot"
    for arg in args:
        if arg.startswith("--nodes=") and not 1 <= int(arg.split("=", 1)[1]) <= 8:
            raise EvidenceError("GPU request exceeds the eight-node limit")
    context = {k: os.environ.get(k) for k in ("RUN", "CFG", "ALPHA", "KD_DIR", "DATA", "MODEL", "OUT",
               "STAGE", "BASE_MODEL", "AUTOPILOT_SAMPLE_SEED", "AUTOPILOT_TRAINING_NOTE")}
    key = hashlib.sha256(json.dumps([args, context], sort_keys=True).encode()).hexdigest()[:20]
    path = directory / "gpu" / ("async-" + key + ".json")
    scheduler = Slurm(config, directory)
    while True:
        if (directory / "STOP").exists():
            raise EvidenceError("STOP is present")
        if (directory / "PAUSE").exists():
            time.sleep(30)
            continue
        with lock(directory / "async_submit.lock") as acquired:
            if not acquired:
                time.sleep(1)
                continue
            state = read_json(path) if path.exists() else {"attempt": 0}
            if state.get("job_id"):
                status, _ = scheduler.status(state["job_id"])
                if status in ACTIVE or status == "UNKNOWN":
                    return state["job_id"]
                if status == "CANCELLED":
                    raise EvidenceError("GPU job was cancelled; automatic resubmission is disabled")
            attempt = state["attempt"] + 1
            job = scheduler.submit(f"async-gpu-{key}-{attempt}", args[-1], gpu_args=args)
            atomic(path, {"job_id": job, "attempt": attempt, "submitted_at": now()})
            return job


if __name__ == "__main__":
    print(submit(sys.argv[1], sys.argv[2:]))
