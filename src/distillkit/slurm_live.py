"""Live Slurm backend for the assistant's tools: the schemas of `distillkit.tools` (the ones round 43 was trained on),
implemented with the real commands (squeue, sacct, sbatch, scancel, sinfo, scontrol) instead of the mock fixtures.
Standard library only.

Commands run locally (on a login node) or on a remote login node through `ssh HOST` (from a laptop). Actions that
change the cluster (submit_job without test_only, cancel_job) go through a `confirm(name, args)` callback first; a
refusal is returned to the model as an error. cancel_job only acts on the user's own jobs.

One tool beyond the trained schemas: `job_script` returns a job's batch script (`scontrol write batch_script`, or
`sacct --batch-script` for finished jobs), so the agent can fix and resubmit (retrigger) a job.
"""
import json
import re
import shlex
import subprocess
import time

from .tools import SCHEMAS, ToolError, parse_arguments

JOB_SCRIPT_SCHEMA = {
    "name": "job_script",
    "description": "The batch script a job was submitted with (to fix and resubmit it with submit_job).",
    "parameters": {"type": "object", "properties": {"job_id": {"type": "string", "description": "Slurm job id"}},
                   "required": ["job_id"]},
}
LIVE_SCHEMAS = SCHEMAS + [JOB_SCRIPT_SCHEMA]
ACTIONS = ("submit_job", "cancel_job")  # change the cluster: confirmed first (submit_job only without test_only)
_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
_TYPES = {"string": str, "integer": int, "boolean": bool}


def openai_tools(include_script=True):
    """The tool list in OpenAI/Ollama format."""
    return [{"type": "function", "function": s} for s in (LIVE_SCHEMAS if include_script else SCHEMAS)]


class SlurmLive:
    def __init__(self, ssh=None, workdir="~/.hpc-agent/jobs", confirm=None, timeout=60, runner=None):
        self.ssh, self.workdir, self.timeout = ssh, workdir, timeout
        self.confirm = confirm or (lambda name, args: False)
        self._runner = runner  # tests: callable(argv, stdin) -> (rc, stdout, stderr)
        self._user = None

    # --- command execution ---------------------------------------------------------------------
    def run(self, argv, stdin=None):
        if self._runner:
            return self._runner(argv, stdin)
        cmd = ["ssh", "-o", "BatchMode=yes", self.ssh, shlex.join(argv)] if self.ssh else argv
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: command not found (is this a Slurm login node?)"
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out after {self.timeout}s"
        return p.returncode, p.stdout, p.stderr

    def user(self):
        if self._user is None:
            rc, out, _ = self.run(["id", "-un"])
            self._user = out.strip() if rc == 0 else ""
        return self._user

    # --- dispatch --------------------------------------------------------------------------------
    def execute(self, name, arguments):
        """Validate and run one call. Raises ToolError for invalid calls; cluster errors come back as {"error": ...}."""
        args = parse_arguments(arguments)
        schema = next((s for s in LIVE_SCHEMAS if s["name"] == name), None)
        if schema is None:
            raise ToolError(f"unknown tool {name!r}")
        props, required = schema["parameters"]["properties"], schema["parameters"]["required"]
        if missing := [p for p in required if p not in args]:
            raise ToolError(f"{name}: missing required argument(s) {missing}")
        if extra := [a for a in args if a not in props]:
            raise ToolError(f"{name}: unknown argument(s) {extra}")
        for a, v in args.items():
            t = _TYPES.get(props[a]["type"])
            if t and (not isinstance(v, t) or (t is int and isinstance(v, bool))):
                raise ToolError(f"{name}: argument {a!r} should be {props[a]['type']}")
            if "enum" in props[a] and v not in props[a]["enum"]:
                raise ToolError(f"{name}: argument {a!r} must be one of {props[a]['enum']}")
        if "job_id" in args and not _JOB_ID.match(args["job_id"]):
            return {"error": f"invalid job id {args['job_id']!r}"}
        if name in ACTIONS and not (name == "submit_job" and args.get("test_only")) and not self.confirm(name, args):
            return {"error": "the user declined this action", "action": name}
        return getattr(self, name)(**args)

    # --- tools -----------------------------------------------------------------------------------
    def job_status(self, job_id):
        fmt = "%i|%j|%T|%r|%P|%D|%M|%l|%N|%S"
        rc, out, err = self.run(["squeue", "-h", "-j", job_id, "-o", fmt])
        line = out.strip().splitlines()[0] if rc == 0 and out.strip() else ""
        if line:
            i, name, state, reason, part, nodes, elapsed, limit, nodelist, start = line.split("|")
            job = {"job_id": i, "name": name, "state": state, "partition": part, "nodes": int(nodes or 0),
                   "elapsed": elapsed, "time_limit": limit}
            if state == "PENDING":
                job |= {"reason": reason, "est_start": start}
            else:
                job["node_list"] = nodelist
            return job
        acct = self._sacct(job_id)  # no longer in the queue: the accounting record
        if acct:
            return {k: acct[k] for k in ("job_id", "name", "state", "partition", "elapsed", "time_limit", "exit_code",
                                          "node_list") if k in acct}
        return {"job_id": job_id, "found": False, "error": (err.strip() or f"job {job_id} not found")}

    def list_queue(self, state=None, partition=None):
        argv = ["squeue", "--me", "-h", "-o", "%i|%j|%T|%P|%M|%l|%r"]
        if state:
            argv += ["-t", state]
        if partition:
            argv += ["-p", partition]
        rc, out, err = self.run(argv)
        if rc != 0:
            return {"error": err.strip() or "squeue failed"}
        jobs = []
        for line in out.strip().splitlines():
            i, name, st, part, elapsed, limit, reason = line.split("|")
            j = {"job_id": i, "name": name, "state": st, "partition": part, "elapsed": elapsed, "time_limit": limit}
            if st == "PENDING":
                j["reason"] = reason
            jobs.append(j)
        return {"jobs": jobs}

    def submit_job(self, script, test_only=False):
        if not script.lstrip().startswith("#!"):
            return {"error": "sbatch: error: This does not look like a batch script. The first line must start with "
                             "#! followed by the path to an interpreter."}
        path = f"{self.workdir.rstrip('/')}/job_{time.strftime('%Y%m%d_%H%M%S')}_{abs(hash(script)) % 10**6:06d}.sh"
        rc, _, err = self.run(["bash", "-c", f'mkdir -p {self.workdir} && cat > {path}'], stdin=script)
        if rc != 0:
            return {"error": f"could not write the script: {err.strip()}"}
        workdir = path.rsplit("/", 1)[0]
        argv = ["bash", "-c", f"cd {workdir} && sbatch --parsable {'--test-only ' if test_only else ''}{path}"]
        rc, out, err = self.run(argv)
        if test_only:  # sbatch --test-only reports on stderr
            msg = (err or out).strip()
            return {"valid": rc == 0, "message": msg, "script_file": path} if rc == 0 else {"error": msg}
        if rc != 0:
            return {"error": err.strip() or out.strip() or "sbatch failed"}
        jid = out.strip().split(";")[0]
        return {"submitted": True, "job_id": jid, "message": f"Submitted batch job {jid}", "script_file": path}

    def cancel_job(self, job_id):
        rc, out, _ = self.run(["squeue", "--me", "-h", "-j", job_id.split("_")[0], "-o", "%i"])
        if rc != 0 or not out.strip():
            return {"cancelled": False, "message": f"job {job_id} is not one of your queued or running jobs"}
        before = self.job_status(job_id).get("state", "unknown")
        rc, _, err = self.run(["scancel", job_id])
        if rc != 0:
            return {"cancelled": False, "message": err.strip() or "scancel failed"}
        return {"cancelled": True, "message": f"Job {job_id} cancelled (was {before})"}

    def read_job_log(self, job_id, stream="stderr", tail_lines=20):
        if not 1 <= tail_lines <= 200:
            raise ToolError("tail_lines must be between 1 and 200")
        path = self._log_path(job_id, stream)
        if not path:
            return {"error": f"no {stream} file found for job {job_id}"}
        rc, out, err = self.run(["tail", "-n", str(tail_lines), path])
        if rc != 0:
            return {"error": err.strip() or f"{path} does not exist (yet)", "file": path}
        res = {"file": path, "lines": out.splitlines()}
        if path == self._log_path(job_id, "stdout" if stream == "stderr" else "stderr"):
            res["note"] = "stdout and stderr go to the same file (no separate --error file)"
        return res

    def job_accounting(self, job_id):
        acct = self._sacct(job_id)
        if not acct:
            return {"job_id": job_id, "error": "no accounting record (yet)"}
        return acct

    def gpu_availability(self, partition=None):
        argv = ["sinfo", "-N", "-h", "-O", "NodeList:40,Partition:40,Gres:80,GresUsed:80,StateCompact:20"]
        if partition:
            argv += ["-p", partition]
        rc, out, err = self.run(argv)
        if rc != 0:
            return {"error": err.strip() or "sinfo failed"}
        parts, seen = {}, set()
        for line in out.splitlines():
            f = line.split()
            if len(f) < 5 or (f[0], f[1]) in seen:
                continue
            seen.add((f[0], f[1]))
            total, gtype = _gpus(f[2])
            if not total:
                continue
            used = _gpus(f[3])[0]
            p = parts.setdefault(f[1].rstrip("*"), {"partition": f[1].rstrip("*"), "gpu_type": gtype, "gpus_total": 0,
                                                    "gpus_free": 0, "idle_nodes": 0})
            p["gpus_total"] += total
            if not re.search(r"drain|down|maint|fail|resv", f[4]):
                p["gpus_free"] += max(0, total - used)
                p["idle_nodes"] += f[4].startswith("idle")
        if partition and not parts:
            return {"error": f"no GPU nodes in partition {partition} (or the partition does not exist)"}
        return {"partitions": list(parts.values())}

    def partition_info(self, partition=None):
        argv = ["scontrol", "show", "partition", "-o"] + ([partition] if partition else [])
        rc, out, err = self.run(argv)
        if rc != 0:
            return {"error": err.strip() or "scontrol failed"}
        res = []
        for line in out.strip().splitlines():
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            name = kv.get("PartitionName")
            rc2, nodes, _ = self.run(["sinfo", "-h", "-p", name, "-o", "%c|%m|%G"])
            cpus = mem = gres = None
            if rc2 == 0 and nodes.strip():
                c, m, g = nodes.strip().splitlines()[0].split("|")
                cpus, mem, gres = int(c.rstrip("+") or 0), f"{int(m.rstrip('+') or 0) // 1024}G", g
            gpus, gtype = _gpus(gres or "")
            res.append({"partition": name, "max_time": kv.get("MaxTime"), "nodes": int(kv.get("TotalNodes", 0)),
                        "max_nodes_per_job": kv.get("MaxNodes"), "qos": kv.get("QoS"), "gpus_per_node": gpus,
                        "gpu_type": gtype, "cpus_per_node": cpus, "mem_per_node": mem})
        if partition:
            return res[0] if res else {"error": f"invalid partition {partition!r}"}
        return {"partitions": res}

    def job_script(self, job_id):
        rc, out, err = self.run(["scontrol", "write", "batch_script", job_id, "-"])
        if rc == 0 and out.strip():
            return {"job_id": job_id, "script": out}
        rc, out, err2 = self.run(["sacct", "-n", "-j", job_id, "--batch-script"])
        if rc == 0 and out.strip().startswith(("#!", "Batch Script")):
            return {"job_id": job_id, "script": re.sub(r"^Batch Script for \S+\n-+\n", "", out)}
        return {"error": (err.strip() or err2.strip() or f"no batch script stored for job {job_id}")}

    # --- helpers ---------------------------------------------------------------------------------
    def _sacct(self, job_id):
        fields = "JobID,JobName,State,ExitCode,Partition,Elapsed,Timelimit,NodeList,MaxRSS,TotalCPU,AllocCPUS,AllocTRES"
        rc, out, _ = self.run(["sacct", "-n", "-P", "-j", job_id, f"--format={fields}"])
        if rc != 0 or not out.strip():
            return None
        rows = [dict(zip(fields.split(","), l.split("|"))) for l in out.strip().splitlines()]
        main = next((r for r in rows if r["JobID"] == job_id), rows[0])
        rss = [_mem_gb(r["MaxRSS"]) for r in rows if r["MaxRSS"]]
        acct = {"job_id": main["JobID"], "name": main["JobName"], "state": main["State"].split()[0],
                "exit_code": main["ExitCode"], "partition": main["Partition"], "elapsed": main["Elapsed"],
                "time_limit": main["Timelimit"], "node_list": main["NodeList"]}
        if rss:
            acct["max_rss"] = f"{max(rss):.1f}G"
        el, cpu, n = _secs(main["Elapsed"]), _secs(main["TotalCPU"]), int(main["AllocCPUS"] or 0)
        if el and n:
            acct["cpu_efficiency"] = f"{100 * cpu / (el * n):.0f}%"
        g = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", main["AllocTRES"])
        acct["gpus"] = int(g.group(1)) if g else 0
        if acct["gpus"]:
            acct["gpu_utilization"] = "not recorded by Slurm accounting on this cluster"
        return acct

    def _log_path(self, job_id, stream):
        key = "StdErr" if stream == "stderr" else "StdOut"
        rc, out, _ = self.run(["scontrol", "show", "job", job_id])
        if rc == 0 and (m := re.search(rf"\b{key}=(\S+)", out)):
            return m.group(1)
        rc, out, _ = self.run(["sacct", "-n", "-P", "-X", "-j", job_id, f"--format={key},WorkDir"])
        if rc == 0 and out.strip():
            path, workdir = (out.strip().splitlines()[0].split("|") + [""])[:2]
            if path:
                return path.replace("%j", job_id.split("_")[0]).replace("%A", job_id.split("_")[0])
            if workdir:
                return f"{workdir}/slurm-{job_id}.out"
        return None


def _gpus(gres):
    """'gpu:a100:4(S:0-1),...' -> (4, 'a100'); used GRES like 'gpu:a100:2(IDX:0-1)' -> (2, 'a100')."""
    total, gtype = 0, None
    for m in re.finditer(r"gpu(?::([A-Za-z0-9_.-]+))?:(\d+)", gres or ""):
        total += int(m.group(2))
        gtype = gtype or m.group(1)
    return total, gtype


def _secs(t):
    """Slurm durations: [D-]HH:MM:SS, MM:SS(.mmm) -> seconds."""
    if not t or t in ("INVALID", "UNLIMITED"):
        return 0
    days, _, rest = t.rpartition("-")
    parts = [float(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _mem_gb(v):
    m = re.match(r"([\d.]+)([KMGT]?)", v or "")
    if not m:
        return 0.0
    return float(m.group(1)) * {"": 1 / 2**30, "K": 1 / 2**20, "M": 1 / 1024, "G": 1, "T": 1024}[m.group(2)]


def describe(name, args):
    """One line for the confirmation prompt."""
    if name == "submit_job":
        head = [l for l in args.get("script", "").splitlines() if l.startswith("#SBATCH")][:6]
        return "submit this job:\n  " + "\n  ".join(head or ["(no #SBATCH lines)"])
    return f"{name} {json.dumps(args)}"
