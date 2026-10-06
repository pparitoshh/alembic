"""HPC tools the assistant can call: JSON schemas + deterministic mock implementations.

The mocks stand in for a real Slurm cluster (Leonardo-like partitions) during data generation,
verification and evaluation. Every result is a pure function of the call, so traces are
reproducible and checkable: the same job id always has the same state, logs and accounting.
The real tools for the demo wrap squeue/sacct/sbatch/etc. behind the same schemas.
"""

import hashlib
import json
import random
import re

from .checks import check_answer, load_flags

_JOB_ID = {"type": "string", "description": "Slurm job id, e.g. '4242' or '4242_7' for an array task"}

SCHEMAS: list[dict] = [
    {
        "name": "job_status",
        "description": "Current state of one Slurm job (like `squeue -j` / `scontrol show job`): state, pending reason, partition, nodes, elapsed time and time limit.",
        "parameters": {"type": "object", "properties": {"job_id": _JOB_ID}, "required": ["job_id"]},
    },
    {
        "name": "list_queue",
        "description": "List the user's queued and running jobs (like `squeue --me`), optionally filtered by state or partition.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["PENDING", "RUNNING"], "description": "Only jobs in this state"},
                "partition": {"type": "string", "description": "Only jobs in this partition"},
            },
            "required": [],
        },
    },
    {
        "name": "submit_job",
        "description": "Submit a batch script (like `sbatch`). With test_only=true, only validate it and estimate the start time (like `sbatch --test-only`).",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Full batch script text, starting with #!/bin/bash and #SBATCH lines"},
                "test_only": {"type": "boolean", "description": "Validate without submitting", "default": False},
            },
            "required": ["script"],
        },
    },
    {
        "name": "cancel_job",
        "description": "Cancel a job or one array task (like `scancel`).",
        "parameters": {"type": "object", "properties": {"job_id": _JOB_ID}, "required": ["job_id"]},
    },
    {
        "name": "read_job_log",
        "description": "Read the last lines of a job's stdout or stderr file.",
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": _JOB_ID,
                "stream": {"type": "string", "enum": ["stdout", "stderr"], "default": "stderr"},
                "tail_lines": {"type": "integer", "description": "Number of lines from the end (1-200)", "default": 20},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "job_accounting",
        "description": "Resource usage of a finished or running job (like `sacct`/`seff`): state, exit code, elapsed time, peak memory, CPU and GPU utilization.",
        "parameters": {"type": "object", "properties": {"job_id": _JOB_ID}, "required": ["job_id"]},
    },
    {
        "name": "gpu_availability",
        "description": "Free and total GPUs per partition right now (like `sinfo` with GRES), optionally for one partition.",
        "parameters": {
            "type": "object",
            "properties": {"partition": {"type": "string", "description": "Partition name; omit for all GPU partitions"}},
            "required": [],
        },
    },
    {
        "name": "partition_info",
        "description": "Limits of a partition or QOS: max wall time, nodes, GPUs and CPUs per node, memory per node.",
        "parameters": {
            "type": "object",
            "properties": {"partition": {"type": "string", "description": "Partition name; omit for all partitions"}},
            "required": [],
        },
    },
]
TOOLS = {s["name"]: s for s in SCHEMAS}

# A Leonardo-like cluster (CINECA docs): Booster nodes = 4x A100 64 GB, 32 cores, 512 GB RAM.
PARTITIONS = {
    "boost_usr_prod": {"max_time": "24:00:00", "nodes": 3456, "gpus_per_node": 4, "gpu_type": "A100-64GB", "cpus_per_node": 32, "mem_per_node": "494G"},
    "boost_qos_dbg": {"max_time": "00:30:00", "nodes": 2, "gpus_per_node": 4, "gpu_type": "A100-64GB", "cpus_per_node": 32, "mem_per_node": "494G"},
    "boost_qos_lprod": {"max_time": "4-00:00:00", "nodes": 3, "gpus_per_node": 4, "gpu_type": "A100-64GB", "cpus_per_node": 32, "mem_per_node": "494G"},
    "dcgp_usr_prod": {"max_time": "24:00:00", "nodes": 1536, "gpus_per_node": 0, "gpu_type": None, "cpus_per_node": 112, "mem_per_node": "494G"},
}
STATES = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED"]
PENDING_REASONS = ["Priority", "Resources", "QOSMaxGRESPerUser", "AssocGrpGRESMinutes", "Dependency", "ReqNodeNotAvail"]

_STDERR = {
    "FAILED": ["Traceback (most recent call last):", '  File "train.py", line 88, in <module>', "    import torch", "ModuleNotFoundError: No module named 'torch'", "srun: error: lrdn0412: task 0: Exited with exit code 1"],
    "OUT_OF_MEMORY": ["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 63.42 GiB of which 1.12 GiB is free.", "slurmstepd: error: Detected 1 oom_kill event in StepId={job}.0. Some of the step tasks have been OOM Killed."],
    "TIMEOUT": ["epoch 37/100 step 11200 loss 0.412", "slurmstepd: error: *** JOB {job} ON lrdn0412 CANCELLED AT 2026-10-12T09:14:02 DUE TO TIME LIMIT ***"],
    "CANCELLED": ["slurmstepd: error: *** JOB {job} ON lrdn0412 CANCELLED AT 2026-10-12T09:14:02 ***"],
}
_STDOUT = ["Using 4 GPUs: A100-SXM-64GB", "epoch 1/10 loss 2.31", "epoch 2/10 loss 1.87", "epoch 3/10 loss 1.52", "checkpoint saved to $SCRATCH/run/ckpt_3.pt"]


class ToolError(ValueError):
    """Invalid call: unknown tool, missing/unknown/mistyped argument."""


def _rng(*key) -> random.Random:
    return random.Random(hashlib.sha256(json.dumps(key).encode()).hexdigest())


def _job(job_id: str) -> dict:
    """The mock cluster's view of a job id: same id -> same job, always."""
    if not str(job_id).replace("_", "").isdigit():
        raise ToolError(f"invalid job id {job_id!r}")
    r = _rng("job", job_id)
    state = r.choice(STATES)
    part = r.choice(["boost_usr_prod"] * 4 + ["boost_qos_dbg", "dcgp_usr_prod"])
    limit = {"boost_qos_dbg": "00:30:00"}.get(part, r.choice(["04:00:00", "12:00:00", "24:00:00"]))
    gpus = 0 if part.startswith("dcgp") else r.choice([1, 2, 4])
    job = {"job_id": job_id, "name": r.choice(["train_llm", "md_sim", "cfd_run", "preprocess", "eval"]), "state": state, "partition": part, "time_limit": limit, "nodes": 1, "gpus": gpus}
    if state == "PENDING":
        job |= {"reason": r.choice(PENDING_REASONS), "elapsed": "00:00:00", "est_start": "2026-10-12T14:30:00"}
    else:
        job |= {"elapsed": limit if state == "TIMEOUT" else _fraction_of(limit, r), "node_list": f"lrdn{r.randint(1, 3456):04d}"}
    if state not in ("PENDING", "RUNNING"):
        job["exit_code"] = {"COMPLETED": "0:0", "FAILED": "1:0", "TIMEOUT": "0:15", "OUT_OF_MEMORY": "0:125", "CANCELLED": "0:15"}[state]
    return job


def _fraction_of(limit: str, r: random.Random) -> str:
    """A random elapsed time strictly inside the time limit (HH:MM:SS)."""
    h, m, s = map(int, limit.split(":"))
    secs = int((h * 3600 + m * 60 + s) * r.uniform(0.05, 0.95))
    return f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}"


def _job_status(job_id: str) -> dict:
    return _job(job_id)


def _list_queue(state: str | None = None, partition: str | None = None) -> dict:
    r = _rng("queue")
    jobs = []
    for jid in sorted({str(r.randint(4000000, 4999999)) for _ in range(8)}):
        j = _job(jid)
        if j["state"] in ("PENDING", "RUNNING"):
            jobs.append({k: j[k] for k in ("job_id", "name", "state", "partition", "elapsed", "time_limit")} | ({"reason": j["reason"]} if "reason" in j else {}))
    jobs = [j for j in jobs if (not state or j["state"] == state) and (not partition or j["partition"] == partition)]
    return {"jobs": jobs}


_VALID_FLAGS: set[str] | None = None  # None = the packaged list, loaded on first use


def set_valid_flags(flags: set[str]) -> None:
    """Use the run's flag list (`verify.flag_list`) so the mock sbatch and the answer checks agree."""
    global _VALID_FLAGS
    _VALID_FLAGS = flags


def _submit_job(script: str, test_only: bool = False) -> dict:
    global _VALID_FLAGS
    if _VALID_FLAGS is None:
        _VALID_FLAGS = load_flags()
    if not script.lstrip().startswith("#!"):
        return {"error": "sbatch: error: This does not look like a batch script. The first line must start with #! followed by the path to an interpreter."}
    res = check_answer(f"```bash\n{script}\n```", _VALID_FLAGS)
    if res["bad_flags"]:
        return {"error": f"sbatch: unrecognized option '--{res['bad_flags'][0]}'"}
    if not res["bash_ok"]:
        return {"error": "sbatch: error: batch script has a shell syntax error"}
    part = re.search(r"^\s*#SBATCH\s+(?:--partition[= ]|-p\s*)(\S+)", script, re.MULTILINE)
    if part and part.group(1) not in PARTITIONS:
        return {"error": "sbatch: error: invalid partition specified: " + part.group(1)}
    partition = part.group(1) if part else "boost_usr_prod"  # this mock's default
    jid = str(4000000 + int(hashlib.sha256(script.encode()).hexdigest(), 16) % 1000000)
    if test_only:
        return {"valid": True, "message": f"sbatch: Job {jid} to start at 2026-10-12T14:30:00 in partition {partition}",
                "validation_scope": ["flag_names", "bash_syntax", "partition_name"]}
    return {"submitted": True, "job_id": jid, "message": f"Submitted batch job {jid}"}


def _cancel_job(job_id: str) -> dict:
    j = _job(job_id)
    if j["state"] not in ("PENDING", "RUNNING"):
        return {"cancelled": False, "message": f"scancel: error: Kill job error on job id {job_id}: Job/step already completing or completed"}
    return {"cancelled": True, "message": f"Job {job_id} cancelled (was {j['state']})"}


def _read_job_log(job_id: str, stream: str = "stderr", tail_lines: int = 20) -> dict:
    if not 1 <= tail_lines <= 200:
        raise ToolError("tail_lines must be between 1 and 200")
    j = _job(job_id)
    if j["state"] == "PENDING":
        return {"error": f"slurm-{job_id}.out does not exist yet: the job has not started"}
    if stream == "stdout":
        lines = _STDOUT
    elif j["state"] == "OUT_OF_MEMORY" and not j["gpus"]:
        lines = _STDERR["OUT_OF_MEMORY"][1:]  # host OOM kill only, no CUDA error
    else:
        lines = _STDERR.get(j["state"], [])
    return {"file": f"slurm-{job_id}.{'out' if stream == 'stdout' else 'err'}", "lines": [l.format(job=job_id) for l in lines][-tail_lines:]}


def _job_accounting(job_id: str) -> dict:
    j = _job(job_id)
    if j["state"] == "PENDING":
        return {"job_id": job_id, "state": "PENDING", "message": "no usage yet: the job has not started"}
    r = _rng("acct", job_id)
    acct = {k: j[k] for k in ("job_id", "state", "elapsed", "time_limit", "partition")} | {"exit_code": j.get("exit_code", "")}
    oom = j["state"] == "OUT_OF_MEMORY"
    # GPU jobs run out of GPU memory (CUDA OOM); CPU jobs hit the node's RAM limit
    acct |= {"max_rss": "494G" if oom and not j["gpus"] else f"{r.randint(4, 300)}G", "cpu_efficiency": f"{r.randint(5, 95)}%"}
    if j["gpus"]:
        acct |= {"gpus": j["gpus"], "gpu_util_mean": f"{r.randint(3, 98)}%", "gpu_mem_peak": "63.4GiB" if oom else f"{r.randint(5, 60)}GiB"}
    return acct


def _gpu_availability(partition: str | None = None) -> dict:
    parts = {p: v for p, v in PARTITIONS.items() if v["gpus_per_node"]}
    if partition:
        if partition not in PARTITIONS:
            return {"error": f"sinfo: error: invalid partition specified: {partition}"}
        parts = {partition: PARTITIONS[partition]} if PARTITIONS[partition]["gpus_per_node"] else {}
    out = []
    for p, v in parts.items():
        r = _rng("gpus", p)
        total = v["nodes"] * v["gpus_per_node"]
        free = r.randint(0, max(1, total // 50))
        # A fully idle node contributes all of its GPUs to this synthetic free count.
        idle = r.randint(0, min(5, v["nodes"], free // v["gpus_per_node"]))
        out.append({"partition": p, "gpu_type": v["gpu_type"], "gpus_total": total, "gpus_free": free, "idle_nodes": idle})
    return {"partitions": out}


def _partition_info(partition: str | None = None) -> dict:
    if partition:
        if partition not in PARTITIONS:
            return {"error": f"invalid partition {partition!r}; known: {sorted(PARTITIONS)}"}
        return {"partition": partition, **PARTITIONS[partition]}
    return {"partitions": [{"partition": p, **v} for p, v in PARTITIONS.items()]}


_IMPL = {
    "job_status": _job_status,
    "list_queue": _list_queue,
    "submit_job": _submit_job,
    "cancel_job": _cancel_job,
    "read_job_log": _read_job_log,
    "job_accounting": _job_accounting,
    "gpu_availability": _gpu_availability,
    "partition_info": _partition_info,
}
assert _IMPL.keys() == TOOLS.keys()

_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float), "object": dict, "array": list}


def parse_arguments(arguments) -> dict:
    """Arguments arrive as a dict (Qwen template / our records) or a JSON string (OpenAI API)."""
    if isinstance(arguments, dict):
        return arguments
    try:
        args = json.loads(arguments or "{}")
    except (TypeError, json.JSONDecodeError) as e:
        raise ToolError(f"arguments are not valid JSON: {e}") from None
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    return args


def validate_call(name: str, args: dict) -> None:
    """Raise ToolError unless `name(**args)` matches the schema: known tool, required args, no extras, types, enums."""
    if name not in TOOLS:
        raise ToolError(f"unknown tool {name!r}")
    params = TOOLS[name]["parameters"]
    props = params["properties"]
    if missing := [p for p in params["required"] if p not in args]:
        raise ToolError(f"{name}: missing required argument(s) {missing}")
    if extra := [a for a in args if a not in props]:
        raise ToolError(f"{name}: unknown argument(s) {extra}")
    for a, v in args.items():
        spec = props[a]
        t = _TYPES[spec["type"]]
        if not isinstance(v, t) or (spec["type"] in ("integer", "number") and isinstance(v, bool)):
            raise ToolError(f"{name}: argument {a!r} should be {spec['type']}, got {type(v).__name__}")
        if "enum" in spec and v not in spec["enum"]:
            raise ToolError(f"{name}: argument {a!r} must be one of {spec['enum']}")


def execute(name: str, arguments) -> dict:
    """Validate and run a call against the mock cluster. Raises ToolError for invalid calls;
    errors the real tool would report (unknown partition, job not started) come back as {"error": ...}."""
    args = parse_arguments(arguments)
    validate_call(name, args)
    return _IMPL[name](**args)
