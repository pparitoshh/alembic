"""Fix 4, step 3 (simulated): teacher trajectories with SIMULATED tool results, so collection needs no live Slurm and
no real jobs.

Each task row gets its own random but consistent world (seeded by the task id): the user's job in the state of the
task's fixture (its random 6-7 digit id replaces {JOB_ID} in the request), other jobs of the user, the partitions,
GPUs, QOS limits and reservations of the cluster. SimSlurm answers the 18 tools of distillkit.slurm_live with the
same schemas (it subclasses SlurmLive, so validation, partition guesses and run_readonly_command filters are the
real code) and the same result shapes. Action tools change the world state and never touch a cluster.

The teacher runs through an OpenAI-compatible endpoint (vLLM serve) with native tool calling, the distillkit system
prompt, thinking off and at most 6 tool calls. Each output row carries the world facts, so a verifier can check the
final answer. Rows have the format research/agent_train/build_agent_data.py reads.

usage: collect_sim.py TASKS.jsonl OUT.jsonl ENDPOINT MODEL [--concurrency N] [--limit N]
"""
import argparse
import copy
import hashlib
import json
import random
import re
import shlex
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from distillkit.assistant import SYSTEM_PROMPT  # noqa: E402
from distillkit.slurm_live import SlurmLive, _looks_like_path, _secs, openai_tools  # noqa: E402
from distillkit.tools import ToolError  # noqa: E402

MAX_CALLS = 6
FIXTURES = (None, "running", "pending", "completed", "failed_oom", "failed_time", "failed_error")
NOW = "2026-10-28T11:20:00"
QUEUED = ("PENDING", "RUNNING")

# ---- cluster facts ---------------------------------------------------------------------------------------------
CLUSTERS = {
    "curiosity": {
        "partitions": {
            "hackathon": {"max_time": "30-12:00:00", "nodes": 16, "max_nodes_per_job": "UNLIMITED", "qos": "eai-team04",
                          "gpus_per_node": 8, "gpu_type": "b300", "cpus_per_node": 256, "mem_per_node": "2048G",
                          "node_prefix": "curiosity-", "node_digits": 3, "default": True},
        },
        "qos": {"name": "eai-team04", "gpu_limit": 8, "cpu_limit": 256, "max_wall": "30-12:00:00"},
        "facts": {"gpu_memory": "~275 GB per B300 GPU", "default_time": "08:00:00", "def_mem_per_cpu": "8G",
                  "debug_qos": None, "containers": "enroot + pyxis (srun --container-image)"},
    },
    "leonardo": {
        "partitions": {
            "boost_usr_prod": {"max_time": "1-00:00:00", "nodes": 3456, "max_nodes_per_job": "UNLIMITED",
                               "qos": "normal", "gpus_per_node": 4, "gpu_type": "a100", "cpus_per_node": 32,
                               "mem_per_node": "481G", "node_prefix": "lrdn", "node_digits": 4, "default": True},
            "dcgp_usr_prod": {"max_time": "1-00:00:00", "nodes": 1536, "max_nodes_per_job": "UNLIMITED",
                              "qos": "normal", "gpus_per_node": 0, "gpu_type": None, "cpus_per_node": 112,
                              "mem_per_node": "481G", "node_prefix": "lrdn", "node_digits": 4},
            "lrd_all_serial": {"max_time": "04:00:00", "nodes": 6, "max_nodes_per_job": "1", "qos": "normal",
                               "gpus_per_node": 0, "gpu_type": None, "cpus_per_node": 4, "mem_per_node": "30G",
                               "node_prefix": "login", "node_digits": 2},
        },
        "qos": {"name": "normal", "gpu_limit": None, "cpu_limit": None, "max_wall": None},
        "facts": {"gpu_memory": "64 GB per A100 GPU", "containers": "Singularity/Apptainer"},
    },
}
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kepler",
         "lumen", "marble", "nimbus", "onyx", "prism", "quartz", "raven", "sierra", "tundra"]
JOB_NAMES = ["train_llm", "finetune", "md_sim", "cfd_run", "preprocess", "eval_bench", "tokenize", "inference"]


def fmt_time(s):
    """Seconds -> Slurm duration [D-]HH:MM:SS."""
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f"{d}-" if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def task_rng(task_id):
    return random.Random(int(hashlib.sha256(str(task_id).encode()).hexdigest()[:16], 16))


# ---- world -----------------------------------------------------------------------------------------------------
def _script(cluster, account, part, name, gpus, nodes, limit, body):
    head = ["#!/bin/bash", f"#SBATCH --job-name={name}", f"#SBATCH --partition={part}", f"#SBATCH --nodes={nodes}",
            f"#SBATCH --time={limit}", f"#SBATCH --output={name}-%j.out"]
    if cluster == "leonardo":
        head.insert(2, f"#SBATCH --account={account}")
    if gpus:
        head.append(f"#SBATCH --gres=gpu:{gpus}")
    return "\n".join(head + [""] + body) + "\n"


def _make_job(rng, w, job_id, state, kind=None, name=None):
    """One job of the user, consistent in every field (state, times, log, accounting)."""
    c = CLUSTERS[w["cluster"]]
    gpu_parts = [p for p, v in c["partitions"].items() if v["gpus_per_node"]]
    part = rng.choice(gpu_parts) if rng.random() < 0.85 else rng.choice(list(c["partitions"]))
    pinfo = c["partitions"][part]
    gpus = rng.choice([g for g in (1, 2, 4, 8) if g <= pinfo["gpus_per_node"]]) if pinfo["gpus_per_node"] else 0
    nodes = 1
    cpus = min(pinfo["cpus_per_node"], max(1, gpus) * (pinfo["cpus_per_node"] // max(1, pinfo["gpus_per_node"] or 1)))
    max_s = _secs(pinfo["max_time"])
    limit_s = min(max_s, rng.choice([1, 2, 4, 8, 12, 24]) * 3600)
    name = name or f"{rng.choice(JOB_NAMES)}_{rng.choice(WORDS)}"
    node = f"{pinfo['node_prefix']}{rng.randint(1, min(pinfo['nodes'], 999)):0{pinfo['node_digits']}d}"
    mem_req = cpus * (8 if w["cluster"] == "curiosity" else 15)
    py = "train.py" if gpus else "run.py"
    if w["cluster"] == "curiosity" and gpus:
        body = [f"srun --container-image=nvcr.io#nvidia/pytorch:25.01-py3 --container-mounts=$PWD:/work "
                f"python /work/{py} --config configs/{name}.yaml"]
    else:
        body = ["module load python/3.11.7", "source $WORK/venvs/main/bin/activate", f"srun python {py} --config configs/{name}.yaml"]
    job = {"job_id": job_id, "name": name, "state": state, "partition": part, "nodes": nodes, "gpus": gpus,
           "cpus": cpus, "memory_requested": f"{mem_req:.1f}G", "time_limit": fmt_time(limit_s), "node_list": node,
           "exit_code": "0:0", "kind": kind, "submit": "2026-10-28T08:12:44",
           "workdir": f"{w['home']}/{name}",
           "script": _script(w["cluster"], w["account"], part, name, gpus, nodes, fmt_time(limit_s), body)}
    job["log_file"] = f"{job['workdir']}/{name}-{job_id}.out"
    el = 0
    start = ["Loading config configs/%s.yaml" % name, f"world size {max(1, gpus)}, rank 0 on {node}"]
    steps = [f"step {k * 100:>6} | loss {3.2 - 0.04 * k:.4f} | lr 2.0e-04 | {0.9 + rng.random():.2f} s/it" for k in range(1, 9)]
    if state == "PENDING":
        job |= {"reason": w.get("pending_reason") or rng.choice(["Priority", "Resources"]), "node_list": "",
                "est_start": "2026-10-28T15:40:00", "log": None}
    elif state == "RUNNING":
        el = int(limit_s * rng.uniform(0.1, 0.7))
        job |= {"log": start + steps, "max_rss": f"{mem_req * rng.uniform(0.2, 0.6):.1f}G"}
    elif state == "COMPLETED":
        el = int(limit_s * rng.uniform(0.2, 0.8))
        job |= {"log": start + steps + ["training done, saved checkpoint to checkpoints/final.pt"],
                "max_rss": f"{mem_req * rng.uniform(0.2, 0.6):.1f}G"}
    elif state == "TIMEOUT":
        el = limit_s
        end = "2026-10-28T10:%02d:%02d" % (rng.randint(0, 59), rng.randint(0, 59))
        job |= {"log": start + steps + [f"slurmstepd: error: *** JOB {job_id} ON {node} CANCELLED AT {end} DUE TO "
                                        "TIME LIMIT ***"], "max_rss": f"{mem_req * 0.4:.1f}G"}
    elif state in ("FAILED", "OUT_OF_MEMORY"):
        el = int(min(limit_s, 3600) * rng.uniform(0.05, 0.5))
        if kind == "cuda_oom":
            tried, total = (rng.choice([12, 20, 38]), 275 if w["cluster"] == "curiosity" else 64)
            tail = ["Traceback (most recent call last):", f'  File "/work/{py}", line 214, in <module>',
                    "    loss = model(**batch).loss", f'  File "/work/{py}", line 131, in forward',
                    "    attn = torch.softmax(scores, dim=-1)",
                    f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {tried}.00 GiB. GPU 0 has a total "
                    f"capacity of {total - 0.4:.2f} GiB of which {rng.uniform(0.2, 3):.2f} GiB is free.",
                    f"srun: error: {node}: task 0: Exited with exit code 1"]
            job |= {"exit_code": "1:0", "max_rss": f"{mem_req * 0.5:.1f}G"}
        elif kind == "host_oom":
            tail = [f"slurmstepd: error: Detected 1 oom-kill event(s) in StepId={job_id}.batch. Some of your "
                    "processes may have been killed by the cgroup out-of-memory handler."]
            job |= {"state": "OUT_OF_MEMORY", "exit_code": "0:125", "max_rss": job["memory_requested"]}
        else:
            mod, data = rng.choice(["flash_attn", "deepspeed", "wandb"]), f"data/{rng.choice(WORDS)}/train.jsonl"
            code, err = rng.choice([
                (f"    rows = open('{data}').readlines()", f"FileNotFoundError: [Errno 2] No such file or directory: '{data}'"),
                (f"    import {mod}", f"ModuleNotFoundError: No module named '{mod}'"),
                ("    lr = cfg['optim']['learning_rate']", "KeyError: 'learning_rate'")])
            tail = ["Traceback (most recent call last):", f'  File "/work/{py}", line 37, in <module>',
                    "    main(load_config(args.config))", f'  File "/work/{py}", line 22, in main', code, err,
                    f"srun: error: {node}: task 0: Exited with exit code 1"]
            job |= {"exit_code": "1:0", "max_rss": f"{mem_req * 0.05:.1f}G", "error_line": err}
        job["log"] = start + (steps[:2] if kind != "host_oom" else steps[:3]) + tail
    job["elapsed"] = fmt_time(el)
    job["elapsed_s"] = el
    job["end"] = "Unknown" if state in QUEUED else f"2026-10-2{rng.randint(6, 8)}T{rng.randint(0, 10):02d}:{rng.randint(0, 59):02d}:00"
    return job


def make_world(task):
    """A random but consistent world for one task (same task id -> same world)."""
    rng = task_rng(task["id"])
    cluster = task["cluster"]
    if cluster not in CLUSTERS:
        raise ValueError(f"unknown cluster {cluster!r}")
    w = {"cluster": cluster, "now": NOW, "jobs": {}, "actions": [], "submitted": []}
    if cluster == "curiosity":
        w |= {"user": f"eai-{''.join(rng.choice('abcdefghjkmnpqrstvwxyz') for _ in range(5))}", "account": "eai-team04"}
        w["home"] = f"/home/{w['user']}"
    else:
        n = rng.randint(1, 9)
        w |= {"user": f"a08tr{rng.choice('abcdefgh')}{rng.randint(10, 99)}", "account": f"tra26_ncc{n}"}
        w["home"] = f"/leonardo_work/{w['account']}/{w['user']}"
        w["reservations"] = [{"name": f"s_tra_ncc{n}", "state": "ACTIVE", "start": "2026-10-28T08:00:00",
                              "end": "2026-10-28T20:00:00", "partition": "boost_usr_prod",
                              "node_count": rng.choice([8, 16, 32]), "accounts": w["account"]}]
    if cluster == "curiosity":
        w["reservations"] = []
    # GPUs in use: the team QOS on Curiosity, the whole partition on both
    c = CLUSTERS[cluster]
    w["partitions"] = {}
    for p, v in c["partitions"].items():
        if not v["gpus_per_node"]:
            w["partitions"][p] = {"gpus_total": 0, "gpus_free": 0, "idle_nodes": rng.randint(0, v["nodes"] // 4)}
            continue
        total = v["gpus_per_node"] * v["nodes"]
        free = rng.choice([0, v["gpus_per_node"], 2 * v["gpus_per_node"], rng.randint(0, total // 4)])
        w["partitions"][p] = {"gpus_total": total, "gpus_free": free, "idle_nodes": free // v["gpus_per_node"]}
    fx = task.get("fixture")
    if fx not in FIXTURES:
        raise ValueError(f"unknown fixture {fx!r}")
    team_used = rng.choice([0, 2, 4, 6, 8]) if cluster == "curiosity" else None
    if fx == "pending" and cluster == "curiosity" and team_used == 8:
        w["pending_reason"] = "QOSGrpGRES"
    w["qos"] = dict(c["qos"])
    w["qos"]["team_gpus_in_use"] = team_used
    w["facts"] = c["facts"]
    # the task's job
    jid = str(rng.randint(100000, 9999999))
    state, kind = {None: ("COMPLETED", None), "running": ("RUNNING", None), "pending": ("PENDING", None),
                   "completed": ("COMPLETED", None), "failed_time": ("TIMEOUT", None),
                   "failed_error": ("FAILED", "python_error"),
                   "failed_oom": ("FAILED", rng.choice(["cuda_oom", "host_oom"]))}[fx]
    w["job_id"] = jid if (fx or "{JOB_ID}" in task.get("user_request", "")) else None
    if w["job_id"]:
        w["jobs"][jid] = _make_job(rng, w, jid, state, kind)
    # other jobs of the user (ids near the task job, some queued, some finished)
    base = int(jid)
    for k in range(rng.randint(1, 4)):
        oid = str(base + rng.choice([-1, 1]) * rng.randint(3, 4000))
        if oid in w["jobs"] or len(oid) < 6:
            continue
        st = rng.choice(["RUNNING", "PENDING", "COMPLETED", "COMPLETED", "FAILED", "CANCELLED"])
        j = _make_job(rng, w, oid, "COMPLETED" if st == "CANCELLED" else st,
                      "python_error" if st == "FAILED" else None)
        if st == "CANCELLED":
            j |= {"state": "CANCELLED", "exit_code": "0:15"}
        w["jobs"][oid] = j
    w["next_id"] = base + 5000
    mine = sum(j["gpus"] for j in w["jobs"].values() if j["state"] == "RUNNING"
               and CLUSTERS[cluster]["partitions"][j["partition"]]["gpus_per_node"])
    w["qos"]["your_gpus_in_use"] = mine
    if team_used is not None:
        w["qos"]["team_gpus_in_use"] = max(team_used, min(8, mine))
    return w


def facts(w):
    """The checkable facts of a world (no logs and scripts): what a verifier compares the final answer with."""
    keep = ("job_id", "name", "state", "partition", "nodes", "gpus", "cpus", "memory_requested", "time_limit",
            "elapsed", "exit_code", "node_list", "reason", "est_start", "max_rss", "kind", "error_line", "log_file")
    jobs = {i: {k: j[k] for k in keep if j.get(k) not in (None, "")} for i, j in w["jobs"].items()}
    parts = {p: {k: v for k, v in CLUSTERS[w["cluster"]]["partitions"][p].items() if k not in ("node_prefix", "node_digits")}
             | w["partitions"][p] for p in w["partitions"]}
    return {"cluster": w["cluster"], "user": w["user"], "account": w["account"], "job_id": w["job_id"], "jobs": jobs,
            "partitions": parts, "qos": w["qos"], "reservations": w["reservations"], "cluster_facts": w["facts"],
            "actions": w["actions"], "submitted": w["submitted"], "now": w["now"]}


# ---- simulator -------------------------------------------------------------------------------------------------
class SimSlurm(SlurmLive):
    """The 18 slurm_live tools on a simulated world. execute() (validation, partition guesses), run_readonly_command
    (refusals, pipe filters) and the action guards are the real SlurmLive code; run() answers the few commands they
    issue from the world state."""

    def __init__(self, world):
        super().__init__(workdir="~/.hpc-agent/jobs", confirm=lambda n, a: True)
        self.w = world
        self.log = []
        self._user = world["user"]
        self._parts = sorted(world["partitions"])

    def execute(self, name, arguments):
        try:
            res = super().execute(name, arguments)
        except ToolError as e:
            self.log.append({"tool": name, "arguments": arguments, "error": str(e)})
            raise
        self.log.append({"tool": name, "arguments": arguments, "result": res})
        return res

    # --- world helpers ---
    def _job(self, job_id):
        return self.w["jobs"].get(str(job_id).split("_")[0])

    def _pinfo(self, part):
        return CLUSTERS[self.w["cluster"]]["partitions"][part]

    def _queued(self):
        return [j for j in self.w["jobs"].values() if j["state"] in QUEUED]

    # --- the commands SlurmLive code issues itself, plus run_readonly_command ---
    def run(self, argv, stdin=None):
        if argv[:1] == ["id"]:
            return 0, self.w["user"] + "\n", ""
        cmd = argv[0] if argv else ""
        if cmd == "squeue":
            jobs = self._queued()
            if "-j" in argv:
                j = self._job(argv[argv.index("-j") + 1])
                jobs = [j] if j and j["state"] in QUEUED else []
            if "-t" in argv:
                jobs = [j for j in jobs if j["state"] == argv[argv.index("-t") + 1].upper()]
            if argv[-1] == "%T":
                return 0, "".join(j["state"] + "\n" for j in jobs), ""
            if argv[-1] == "%i":
                return 0, "".join(j["job_id"] + "\n" for j in jobs), ""
            out = "             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)\n"
            for j in jobs:
                where = f"({j['reason']})" if j["state"] == "PENDING" else j["node_list"]
                out += (f"{j['job_id']:>18} {j['partition'][:9]:>9} {j['name'][:8]:>8} {self.w['user'][:8]:>8} "
                        f"{j['state'][:2]:>2} {j['elapsed']:>10} {j['nodes']:>6} {where}\n")
            return 0, out, ""
        if cmd == "sinfo":
            out = "PARTITION        AVAIL  TIMELIMIT  NODES  STATE NODELIST\n"
            for p, v in CLUSTERS[self.w["cluster"]]["partitions"].items():
                idle = self.w["partitions"][p]["idle_nodes"]
                out += f"{p + ('*' if v.get('default') else ''):<16} up {v['max_time']:>12} {v['nodes'] - idle:>6}    mix {v['node_prefix']}[...]\n"
                out += f"{p + ('*' if v.get('default') else ''):<16} up {v['max_time']:>12} {idle:>6}   idle {v['node_prefix']}[...]\n"
            return 0, out, ""
        if cmd == "sacct":
            out = "JobID           JobName  Partition    Account  AllocCPUS      State ExitCode \n"
            out += "------------ ---------- ---------- ---------- ---------- ---------- -------- \n"
            jobs = list(self.w["jobs"].values())
            if "-j" in argv:
                jobs = [j for j in jobs if j["job_id"] in argv[argv.index("-j") + 1].split(",")]
            for j in jobs:
                out += (f"{j['job_id']:<12} {j['name'][:10]:>10} {j['partition'][:10]:>10} {self.w['account'][:10]:>10} "
                        f"{j['cpus']:>10} {j['state'][:10]:>10} {j['exit_code']:>8} \n")
            return 0, out, ""
        if cmd == "sacctmgr":
            q = self.w["qos"]
            if "qos" in argv[argv.index("show") + 1:argv.index("show") + 2]:
                grp = ",".join(x for x in ([f"cpu={q['cpu_limit']}"] if q["cpu_limit"] else []) +
                               ([f"gres/gpu={q['gpu_limit']}"] if q["gpu_limit"] else []))
                return 0, f"{q['name']}|{grp}||{q['max_wall'] or ''}\n", ""
            return 0, f"{self.w['account']}|{self.w['user']}||{q['name']}\n", ""
        if cmd == "sshare":
            return 0, self._sshare(), ""
        if cmd == "sprio":
            return 0, self._sprio(argv[argv.index("-j") + 1] if "-j" in argv else None), ""
        if cmd == "sstat":
            return 0, "", ""
        if argv[:3] == ["scontrol", "show", "job"] and len(argv) > 3:
            j = self._job(argv[3])
            if not j:
                return 1, "", "slurm_load_jobs error: Invalid job id specified"
            return 0, (f"JobId={j['job_id']} JobName={j['name']} UserId={self.w['user']} JobState={j['state']} "
                       f"Partition={j['partition']} TimeLimit={j['time_limit']} RunTime={j['elapsed']} "
                       f"NumNodes={j['nodes']} NodeList={j['node_list'] or '(null)'} StdOut={j['log_file']} "
                       f"StdErr={j['log_file']}\n"), ""
        if argv[:3] == ["scontrol", "show", "partition"]:
            return 0, "".join(f"PartitionName={p} MaxTime={v['max_time']} TotalNodes={v['nodes']} QoS={v['qos']}\n"
                              for p, v in CLUSTERS[self.w["cluster"]]["partitions"].items()), ""
        if argv[:3] == ["scontrol", "show", "reservation"]:
            if not self.w["reservations"]:
                return 0, "No reservations in the system\n", ""
            return 0, "\n\n".join(f"ReservationName={r['name']} StartTime={r['start']} EndTime={r['end']} "
                                  f"PartitionName={r['partition']} NodeCnt={r['node_count']} Accounts={r['accounts']} "
                                  f"State={r['state']}" for r in self.w["reservations"]) + "\n", ""
        if argv[:2] == ["scontrol", "show"]:
            return 0, f"{' '.join(argv[2:])}: no further details in this view\n", ""
        return 1, "", f"{cmd}: command not available"

    def _sshare(self):
        return ("             Account       User  RawShares  NormShares    RawUsage  EffectvUsage  FairShare \n"
                "-------------------- ---------- ---------- ----------- ----------- ------------- ---------- \n"
                f"{self.w['account']:>20} {self.w['user'][:10]:>10}          1    0.010000     1843211      0.021000   0.412000 \n")

    def _sprio(self, job_id):
        head = "          JOBID PARTITION     USER   PRIORITY       SITE        AGE  FAIRSHARE    JOBSIZE        QOS\n"
        j = self._job(job_id) if job_id else None
        if not j or j["state"] != "PENDING":
            return head
        return head + f"{j['job_id']:>15} {j['partition'][:9]:>9} {self.w['user'][:8]:>8}     104512          0       1200      41200       2112      60000\n"

    # --- tools ---
    def job_status(self, job_id):
        j = self._job(job_id)
        if not j:
            return {"job_id": job_id, "found": False, "error": "slurm_load_jobs error: Invalid job id specified"}
        if j["state"] in QUEUED:
            res = {k: j[k] for k in ("job_id", "name", "state", "partition", "nodes", "elapsed", "time_limit")}
            if j["state"] == "PENDING":
                return res | {"reason": j["reason"], "est_start": j.get("est_start", "N/A")}
            return res | {"node_list": j["node_list"]}
        return {k: j[k] for k in ("job_id", "name", "state", "partition", "elapsed", "time_limit", "exit_code", "node_list")}

    def list_queue(self, state=None, partition=None):
        jobs = []
        for j in self._queued():
            if (state and j["state"] != state) or (partition and j["partition"] != partition):
                continue
            r = {k: j[k] for k in ("job_id", "name", "state", "partition", "elapsed", "time_limit")}
            if j["state"] == "PENDING":
                r["reason"] = j["reason"]
            jobs.append(r)
        if not jobs:
            return {"jobs": [], "message": f"You have no {state} jobs." if state else "You have no running or pending jobs."}
        return {"jobs": jobs}

    def submit_job(self, script, test_only=False):
        if not script.lstrip().startswith("#!"):
            if _looks_like_path(script):
                return {"error": "script must be the full text of the batch script, starting with #! and #SBATCH "
                                 "lines. A path or URL does not work: the server cannot read your files. "
                                 "Ask the user for the script text."}
            return {"error": "sbatch: error: This does not look like a batch script. The first line must start with "
                             "#! followed by the path to an interpreter."}
        opts = _sbatch_opts(script)
        c = CLUSTERS[self.w["cluster"]]
        part = opts.get("partition") or next(p for p, v in c["partitions"].items() if v.get("default"))
        if part not in c["partitions"]:
            return {"error": "sbatch: error: invalid partition specified: " + part}
        pinfo = c["partitions"][part]
        if self.w["cluster"] == "leonardo" and not opts.get("account"):
            return {"error": "sbatch: error: You must specify an account (--account/-A) to submit jobs\n"
                             "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"}
        limit = opts.get("time") or c["facts"].get("default_time") or pinfo["max_time"]
        if _secs(_norm_time(limit)) > _secs(pinfo["max_time"]):
            return {"error": "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)"}
        gpn = _gpus_per_node(opts)
        if gpn > pinfo["gpus_per_node"]:
            return {"error": "sbatch: error: Batch job submission failed: Requested node configuration is not available"}
        nodes = int(re.match(r"\d+", opts.get("nodes", "1")).group()) if re.match(r"\d+", opts.get("nodes", "1")) else 1
        jid = str(self.w["next_id"])
        path = f"{self.workdir}/job_20261028_112000_{int(hashlib.sha256(script.encode()).hexdigest()[:6], 16) % 10**6:06d}.sh"
        if test_only:
            node = f"{pinfo['node_prefix']}{1:0{pinfo['node_digits']}d}"
            cpus = int(opts.get("cpus-per-task", "1")) if opts.get("cpus-per-task", "1").isdigit() else 1
            return {"valid": True, "script_file": path,
                    "message": f"sbatch: Job {jid} to start at 2026-10-28T11:20:05 using {cpus * nodes} processors on "
                               f"nodes {node} in partition {part}"}
        self.w["next_id"] += 1
        name = opts.get("job-name", Path(path).name)
        j = {"job_id": jid, "name": name, "state": "PENDING", "partition": part, "nodes": nodes, "gpus": gpn * nodes,
             "cpus": 1, "memory_requested": "8.0G", "time_limit": fmt_time(_secs(_norm_time(limit))), "node_list": "",
             "exit_code": "0:0", "kind": None, "reason": "Priority", "est_start": "2026-10-28T13:05:00", "log": None,
             "elapsed": "00:00:00", "elapsed_s": 0, "end": "Unknown", "workdir": self.workdir, "script": script,
             "log_file": f"{self.workdir}/slurm-{jid}.out", "submit": NOW}
        if self.w["cluster"] == "curiosity" and gpn and (self.w["qos"]["team_gpus_in_use"] or 0) + gpn * nodes > 8:
            j["reason"] = "QOSGrpGRES"
        self.w["jobs"][jid] = j
        self.w["submitted"].append(jid)
        self.w["actions"].append({"action": "submit_job", "job_id": jid, "partition": part})
        return {"submitted": True, "job_id": jid, "message": f"Submitted batch job {jid}", "script_file": path}

    def cancel_job(self, job_id):
        j = self._job(job_id)
        if not j or j["state"] not in QUEUED:
            return {"cancelled": False, "message": f"job {job_id} is not one of your queued or running jobs"}
        before = j["state"]
        j |= {"state": "CANCELLED", "exit_code": "0:15", "end": NOW}
        self.w["actions"].append({"action": "cancel_job", "job_id": j["job_id"], "before": before})
        return {"cancelled": True, "message": f"Job {job_id} cancelled (was {before})"}

    def read_job_log(self, job_id, stream="stderr", tail_lines=20):
        if not 1 <= tail_lines <= 200:
            raise ToolError("tail_lines must be between 1 and 200")
        j = self._job(job_id)
        if not j:
            return {"error": f"no {stream} file found for job {job_id}"}
        if not j.get("log"):
            return {"error": f"{j['log_file']} does not exist (yet)", "file": j["log_file"]}
        return {"file": j["log_file"], "lines": j["log"][-tail_lines:],
                "note": "stdout and stderr go to the same file (no separate --error file)"}

    def job_accounting(self, job_id):
        j = self._job(job_id)
        if not j:
            return {"job_id": job_id, "error": "no accounting record (yet)"}
        acct = {k: j[k] for k in ("job_id", "name", "state", "exit_code", "partition", "elapsed", "time_limit", "node_list")}
        if j.get("max_rss"):
            acct["max_rss"] = j["max_rss"]
        acct["memory_requested"] = j["memory_requested"]
        req = float(j["memory_requested"][:-1])
        if j.get("max_rss") and req:
            acct["memory_efficiency"] = f"{100 * float(j['max_rss'][:-1]) / req:.0f}%"
        if j["elapsed_s"]:
            eff = {"FAILED": 35, "OUT_OF_MEMORY": 60}.get(j["state"], 85) if j["gpus"] else 92
            acct["cpu_efficiency"] = f"{eff}%"
            acct["time_limit_used"] = f"{100 * j['elapsed_s'] / _secs(j['time_limit']):.0f}%"
        acct["gpus"] = j["gpus"]
        if j["gpus"]:
            acct["gpu_utilization"] = "not recorded by Slurm accounting on this cluster"
        return acct

    def gpu_availability(self, partition=None):
        parts, by_type = [], {}
        for p, v in CLUSTERS[self.w["cluster"]]["partitions"].items():
            if not v["gpus_per_node"] or (partition and p != partition):
                continue
            s = self.w["partitions"][p]
            parts.append({"partition": p, "gpu_type": v["gpu_type"], "gpus_total": s["gpus_total"],
                          "gpus_free": s["gpus_free"], "idle_nodes": s["idle_nodes"], "allowed_for_you": True})
            by_type[v["gpu_type"]] = by_type.get(v["gpu_type"], 0) + s["gpus_free"]
        if partition and not parts:
            return {"error": f"no GPU nodes in partition {partition} (or the partition does not exist)"}
        return {"partitions": parts, "free_gpus_by_type": by_type, "your_limits": self._limits()}

    def _limits(self):
        q = self.w["qos"]
        lim = {"qos": q["name"], "gpu_limit": q["gpu_limit"], "cpu_limit": q["cpu_limit"], "max_wall": q["max_wall"],
               "team_gpus_in_use": q["team_gpus_in_use"], "your_gpus_in_use": q["your_gpus_in_use"],
               "gpus_free_in_team_limit": None}
        if q["gpu_limit"] is None:
            lim["note"] = "no group GPU limit in this QOS"
            return lim
        free = max(0, q["gpu_limit"] - q["team_gpus_in_use"])
        lim["gpus_free_in_team_limit"] = free
        if free == 0:
            lim["note"] = (f"Your team uses all {q['gpu_limit']} GPUs of QOS {q['name']} now ({q['your_gpus_in_use']} of "
                           "them are yours). Slurm still accepts a new GPU job, but it waits PENDING (reason QOSGrpGRES "
                           "or QOSGrpGpuLimit) until team GPUs are free. sbatch --test-only does not check this limit.")
        else:
            lim["note"] = f"{free} of {q['gpu_limit']} team GPUs are free now. The limit is shared by your team."
        return lim

    def partition_info(self, partition=None):
        res = []
        for p, v in CLUSTERS[self.w["cluster"]]["partitions"].items():
            if partition and p != partition:
                continue
            res.append({"partition": p, "max_time": v["max_time"], "nodes": v["nodes"],
                        "max_nodes_per_job": v["max_nodes_per_job"], "qos": v["qos"], "gpus_per_node": v["gpus_per_node"],
                        "gpu_type": v["gpu_type"], "cpus_per_node": v["cpus_per_node"], "mem_per_node": v["mem_per_node"]})
        if partition:
            return res[0] if res else {"error": f"invalid partition {partition!r}"}
        return {"partitions": res}

    def reservation_info(self, name=None):
        found = [r for r in self.w["reservations"] if not name or r["name"] == name]
        if name and not found:
            return {"error": f"Reservation {name} not found"}
        out = []
        for r in found:
            jobs = [{"job_id": j["job_id"], "name": j["name"], "user": self.w["user"], "state": j["state"],
                     "time": j["elapsed"]} for j in self._queued() if j["partition"] == r["partition"]]
            out.append(dict(r) | {"jobs": jobs})
        return {"reservations": out}

    def job_script(self, job_id):
        j = self._job(job_id)
        if not j:
            return {"job_id": job_id, "error": f"no job {job_id} found"}
        return {"job_id": job_id, "script": j["script"]}

    def recent_jobs(self, days=7):
        rows = sorted(self.w["jobs"].values(), key=lambda j: "9999" if j["end"] == "Unknown" else j["end"], reverse=True)
        jobs = [{"job_id": j["job_id"], "name": j["name"], "state": j["state"], "exit_code": j["exit_code"],
                 "elapsed": j["elapsed"], "end": j["end"], "partition": j["partition"]} for j in rows[:10]]
        return {"jobs": jobs, "days": days} if jobs else {"jobs": [], "note": f"no jobs in the last {days} days"}

    def job_usage(self, job_id):
        j = self._job(job_id)
        if not j or j["state"] != "RUNNING":
            return {"error": f"no live usage for job {job_id}: it is not running (sstat only covers running jobs). "
                             "For finished jobs use job_accounting."}
        rss = float(j["max_rss"][:-1])
        return {"job_id": job_id, "steps": [{"JobID": f"{j['job_id']}.batch", "AveCPU": fmt_time(j["elapsed_s"] * 0.8),
                                             "MaxRSS": f"{rss:.1f}G", "AveRSS": f"{rss * 0.8:.1f}G",
                                             "MaxVMSize": f"{rss * 1.6:.1f}G"}]}

    def job_priority(self, job_id):
        out = self._sprio(job_id).strip()
        res = {"job_id": job_id, "priority": out, "sprio_exit_code": 0}
        if not any(l.split() and l.split()[0].isdigit() for l in out.splitlines()):
            res["note"] = f"job {job_id} is not pending (sprio only ranks waiting jobs)"
        return res | {"shares": self._sshare().strip(), "sshare_exit_code": 0}

    def update_job(self, job_id, field, value):
        if re.search(r"[\s|;&><$`()\\]", value) or value.startswith("-"):
            return {"error": f"invalid value {value!r} (no whitespace or shell characters allowed)"}
        j = self._job(job_id)
        if not j or j["state"] not in QUEUED:
            return {"updated": False, "message": f"job {job_id} is not one of your queued or running jobs"}
        parts = CLUSTERS[self.w["cluster"]]["partitions"]
        if field == "TimeLimit":
            new = _secs(_norm_time(value))
            if not new or new > _secs(parts[j["partition"]]["max_time"]):
                return {"updated": False, "message": "Requested time limit is invalid (missing or exceeds some limit)"}
            if j["state"] == "RUNNING" and new > _secs(j["time_limit"]):
                return {"updated": False, "message": "Access/permission denied"}
            j["time_limit"] = fmt_time(new)
        elif field == "Partition":
            if value not in parts:
                return {"updated": False, "message": "Invalid partition name specified"}
            if j["state"] == "RUNNING":
                return {"updated": False, "message": "Job is no longer pending execution"}
            j["partition"] = value
        elif field == "NumNodes":
            if j["state"] == "RUNNING" or not value.isdigit():
                return {"updated": False, "message": "Job is no longer pending execution" if j["state"] == "RUNNING"
                        else "Invalid node count specified"}
            j["nodes"] = int(value)
        else:
            if value not in (self.w["qos"]["name"],):
                return {"updated": False, "message": "Invalid qos specification"}
        self.w["actions"].append({"action": "update_job", "job_id": j["job_id"], "field": field, "value": value})
        return {"updated": True, "job_id": job_id, "field": field, "value": value}

    def _scontrol_act(self, action, job_id):
        j = self._job(job_id)
        if not j or (j["state"] not in QUEUED and action != "requeue") or j["state"] == "CANCELLED":
            return {action: False, "message": f"job {job_id} is not one of your queued or running jobs"}
        past = {"hold": "held", "release": "released", "requeue": "requeued"}[action]
        if action == "hold":
            if j["state"] != "PENDING":
                return {action: False, "message": "Job is no longer pending execution"}
            j["reason"] = "JobHeldUser"
        elif action == "release":
            if j.get("reason") not in ("JobHeldUser", "JobHeldAdmin"):
                return {action: False, "message": "Job is not held"}
            j["reason"] = "Priority"
        else:
            j |= {"state": "PENDING", "reason": "BeginTime", "node_list": "", "elapsed": "00:00:00", "elapsed_s": 0,
                  "end": "Unknown", "est_start": "2026-10-28T11:22:00", "exit_code": "0:0", "log": None}
        self.w["actions"].append({"action": f"{action}_job", "job_id": j["job_id"]})
        return {action: True, "job_id": job_id, "message": f"Job {job_id} {past}"}


def _norm_time(t):
    """sbatch time forms (MM, MM:SS, HH:MM:SS, D-HH[:MM[:SS]]) -> [D-]HH:MM:SS."""
    t = str(t).strip()
    if "-" in t:
        d, rest = t.split("-", 1)
        rest = (rest.split(":") + ["00", "00"])[:3]
        return f"{d}-" + ":".join(rest)
    p = t.split(":")
    if len(p) == 1:
        return fmt_time(int(p[0]) * 60) if p[0].isdigit() else t
    if len(p) == 2:
        return f"00:{p[0]}:{p[1]}"
    return t


_SHORT = {"p": "partition", "t": "time", "A": "account", "N": "nodes", "J": "job-name", "c": "cpus-per-task", "G": "gpus"}


def _sbatch_opts(script):
    """{option: value} of the script's #SBATCH lines (before the first command)."""
    opts = {}
    for line in script.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            break
        if not s.upper().startswith("#SBATCH"):
            continue
        toks = shlex.split(s[7:], comments=False) if s.count('"') % 2 == 0 else s[7:].split()
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok.startswith("--"):
                k, _, v = tok[2:].partition("=")
                if not v and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                    v, i = toks[i + 1], i + 1
                opts[k] = v
            elif tok.startswith("-") and len(tok) > 1 and tok[1] in _SHORT:
                v = tok[2:]
                if not v and i + 1 < len(toks):
                    v, i = toks[i + 1], i + 1
                opts[_SHORT[tok[1]]] = v
            i += 1
    return opts


def _gpus_per_node(opts):
    for k in ("gres", "gpus-per-node", "gpus"):
        if opts.get(k):
            m = re.search(r"(\d+)$", opts[k])
            return int(m.group(1)) if m else 1
    return 0


# ---- teacher loop ----------------------------------------------------------------------------------------------
def _dump_call(tc):
    f = tc.function
    return {"id": tc.id, "type": "function", "function": {"name": f.name, "arguments": f.arguments or "{}"}}


def run_task(client, model, task, max_calls=MAX_CALLS):
    """One trajectory: the teacher solves the task on its simulated world. Returns the output row."""
    world = make_world(task)
    question = task["user_request"].replace("{JOB_ID}", world["job_id"] or "")
    sim = SimSlurm(world)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    tools, n_calls, error, capped = openai_tools(), 0, None, False
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        while True:
            force_answer = n_calls >= max_calls
            resp = client.chat.completions.create(model=model, messages=messages, tools=tools, temperature=0,
                                                  tool_choice="none" if force_answer else "auto", extra_body=extra)
            msg = resp.choices[0].message
            calls = list(msg.tool_calls or [])
            if not calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                break
            if force_answer:
                capped, error = True, f"more than {max_calls} tool calls"
                break
            calls = calls[:max_calls - n_calls]
            capped = capped or len(msg.tool_calls) > len(calls)
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [_dump_call(tc) for tc in calls]})
            for tc in calls:
                n_calls += 1
                try:
                    result = sim.execute(tc.function.name, tc.function.arguments or "{}")
                except ToolError as e:
                    result = {"error": str(e)}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    except Exception as e:  # a server error: keep the row, the verifier drops it
        error = repr(e)
    return {"id": task["id"], "family": task["family"], "question": question, "pass": None, "tools": sim.log,
            "messages": messages, "world": facts(world), "task": task, "n_tool_calls": n_calls, "capped": capped,
            "error": error}


def done_ids(out_path):
    p = Path(out_path)
    if not p.exists():
        return set()
    ids = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["id"])
        except (ValueError, KeyError):
            continue  # a line cut by a killed run
    return ids


def collect(client, model, tasks, out_path, concurrency=8, limit=None):
    """Run the tasks not yet in out_path; append one row per task. Returns the number of new rows."""
    done = done_ids(out_path)
    todo = [t for t in tasks if t["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()

    def one(task):
        row = run_task(client, model, copy.deepcopy(task))
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        rows = list(ex.map(one, todo))
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Teacher trajectories with simulated Slurm tool results.")
    ap.add_argument("tasks")
    ap.add_argument("out")
    ap.add_argument("endpoint")
    ap.add_argument("model")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from openai import OpenAI
    client = OpenAI(base_url=a.endpoint, api_key="EMPTY", timeout=600)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    n = collect(client, a.model, tasks, a.out, a.concurrency, a.limit)
    print(f"collected {n} new trajectories -> {a.out}")


if __name__ == "__main__":
    main()
