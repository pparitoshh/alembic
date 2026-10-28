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
JOB_USAGE_SCHEMA = {
    "name": "job_usage",
    "description": "Live resource usage of a running job: CPU time and memory (current and peak).",
    "parameters": {"type": "object", "properties": {"job_id": {"type": "string", "description": "Slurm job id"}},
                   "required": ["job_id"]},
}
JOB_PRIORITY_SCHEMA = {
    "name": "job_priority",
    "description": "Why a job is waiting: priority factors (age, fairness, size, QOS) and the shares behind them.",
    "parameters": {"type": "object", "properties": {"job_id": {"type": "string", "description": "Slurm job id"}},
                   "required": ["job_id"]},
}
UPDATE_JOB_SCHEMA = {
    "name": "update_job",
    "description": "Change one setting of a pending job (time limit, partition, node count or QOS).",
    "parameters": {"type": "object", "properties": {
        "job_id": {"type": "string", "description": "Slurm job id"},
        "field": {"type": "string", "enum": ["TimeLimit", "Partition", "NumNodes", "QOS"], "description": "Field to change"},
        "value": {"type": "string", "description": "New value for the field"}},
        "required": ["job_id", "field", "value"]},
}
HOLD_JOB_SCHEMA = {
    "name": "hold_job",
    "description": "Hold a pending job so the scheduler does not start it (release it later).",
    "parameters": {"type": "object", "properties": {"job_id": {"type": "string", "description": "Slurm job id"}},
                   "required": ["job_id"]},
}
RELEASE_JOB_SCHEMA = {
    "name": "release_job",
    "description": "Release a held job so the scheduler can start it again.",
    "parameters": {"type": "object", "properties": {"job_id": {"type": "string", "description": "Slurm job id"}},
                   "required": ["job_id"]},
}
REQUEUE_JOB_SCHEMA = {
    "name": "requeue_job",
    "description": "Put a finished or running job back in the queue to run again.",
    "parameters": {"type": "object", "properties": {"job_id": {"type": "string", "description": "Slurm job id"}},
                   "required": ["job_id"]},
}
RUN_READONLY_COMMAND_SCHEMA = {
    "name": "run_readonly_command",
    "description": "Run a read-only Slurm command (squeue, sinfo, sacct, sstat, sprio, sshare, sacctmgr show, scontrol show) and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The read-only Slurm command to run"}},
                   "required": ["command"]},
}
LIVE_SCHEMAS = SCHEMAS + [JOB_SCRIPT_SCHEMA, JOB_USAGE_SCHEMA, JOB_PRIORITY_SCHEMA, UPDATE_JOB_SCHEMA,
                          HOLD_JOB_SCHEMA, RELEASE_JOB_SCHEMA, REQUEUE_JOB_SCHEMA, RUN_READONLY_COMMAND_SCHEMA]
ACTIONS = ("submit_job", "cancel_job", "update_job", "hold_job", "release_job", "requeue_job")
# change the cluster: confirmed first (submit_job only without test_only)
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
        self._mux = None
        self._parts = None

    # --- command execution ---------------------------------------------------------------------
    def run(self, argv, stdin=None):
        if self._runner:
            return self._runner(argv, stdin)
        cmd = ["ssh", "-o", "BatchMode=yes", *self.mux_options(), self.ssh, shlex.join(argv)] if self.ssh else argv
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            return 127, "", f"{argv[0]}: command not found (is this a Slurm login node?)"
        except subprocess.TimeoutExpired:
            return 124, "", f"{argv[0]}: timed out after {self.timeout}s"
        return p.returncode, p.stdout, p.stderr

    def mux_options(self):
        """Reuse one ssh connection for all tool calls (a new connection through a jump host can take 20 s).
        Hosts that already have a ControlPath in the user's ssh config keep their own setting."""
        if self._mux is None:
            try:
                conf = subprocess.run(["ssh", "-G", self.ssh], capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.TimeoutExpired):
                conf = ""
            own = re.search(r"^controlpath (?!none$)", conf, re.M | re.I)
            self._mux = [] if own or not conf else [
                "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/clusterpilot-%C", "-o", "ControlPersist=10m"]
        return self._mux

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
        note = None
        if args.get("partition"):
            args["partition"], note = self.resolve_partition(args["partition"], all_ok=name == "gpu_availability")
            if args["partition"] is False:
                return {"error": note}
            if args["partition"] is None:
                del args["partition"]
        result = getattr(self, name)(**args)
        if isinstance(result, dict) and "invalid partition" in str(result.get("error", "")).lower() and self.partitions():
            result["error"] += "; partitions on this cluster: " + ", ".join(self.partitions())
        if note and isinstance(result, dict):
            result["note"] = note
        return result

    def partitions(self):
        if self._parts is None:
            rc, out, _ = self.run(["sinfo", "-h", "-o", "%R"])
            self._parts = sorted(set(out.split())) if rc == 0 else []
        return self._parts

    def resolve_partition(self, name, all_ok=False):
        """Models often guess a partition ("boost", "gpu", the cluster's name). Returns (partition, note): the exact
        name; the one partition that contains the guess; None (all partitions) if all_ok; or False with an error that
        lists the real partitions, so the model can retry."""
        parts = self.partitions()
        if not parts or name in parts:
            return name, None
        match = [p for p in parts if name.lower() in p.lower()]
        if len(match) == 1:
            return match[0], f"there is no partition {name!r}; used {match[0]!r}"
        listed = ", ".join(parts)
        if all_ok:
            return None, f"{name!r} is not a partition name, so all partitions are shown (partitions: {listed})"
        return False, f"there is no partition {name!r}; partitions on this cluster: {listed}"

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

    # --- own-jobs guard (same as cancel_job) -------------------------------------------------------
    def _own_job(self, job_id):
        """True if the job is one of the user's queued or running jobs."""
        rc, out, _ = self.run(["squeue", "--me", "-h", "-j", job_id.split("_")[0], "-o", "%i"])
        return rc == 0 and bool(out.strip())

    def job_usage(self, job_id):
        fields = ["JobID", "AveCPU", "MaxRSS", "AveRSS", "MaxVMSize"]
        rc, out, _ = self.run(["sstat", "-j", job_id, f"--format={','.join(fields)}", "-P", "-n"])
        rows = [dict(zip(fields, l.split("|"))) for l in out.strip().splitlines() if l.strip()] if rc == 0 else []
        if not rows:
            return {"error": f"sstat has no live usage for job {job_id} (it only covers running jobs). "
                             "For finished jobs use job_accounting."}
        return {"job_id": job_id, "steps": rows}

    def job_priority(self, job_id):
        rc, out, err = self.run(["sprio", "-j", job_id, "-l"])
        if rc != 0:
            return {"error": err.strip() or "sprio failed"}
        res = {"job_id": job_id, "priority": out.strip(), "sprio_exit_code": rc}
        if not any(l.split() and l.split()[0].isdigit() for l in out.strip().splitlines()):
            res["note"] = f"job {job_id} is not pending (sprio only ranks waiting jobs)"
        rc, out, err = self.run(["sshare", "-U", "-l"])
        res |= {"shares": out.strip(), "sshare_exit_code": rc}
        if rc != 0:
            res["shares_error"] = err.strip() or "sshare failed"
        return res

    def update_job(self, job_id, field, value):
        if re.search(r"[\s|;&><$`()\\]", value) or value.startswith("-"):
            return {"error": f"invalid value {value!r} (no whitespace or shell characters allowed)"}
        if not self._own_job(job_id):
            return {"updated": False, "message": f"job {job_id} is not one of your queued or running jobs"}
        rc, out, err = self.run(["scontrol", "update", f"JobId={job_id}", f"{field}={value}"])
        if rc != 0:
            return {"updated": False, "message": err.strip() or "scontrol update failed"}
        return {"updated": True, "job_id": job_id, "field": field, "value": value}

    def hold_job(self, job_id):
        return self._scontrol_act("hold", job_id)

    def release_job(self, job_id):
        return self._scontrol_act("release", job_id)

    def requeue_job(self, job_id):
        return self._scontrol_act("requeue", job_id)

    def _scontrol_act(self, action, job_id):
        if not self._own_job(job_id):
            return {action: False, "message": f"job {job_id} is not one of your queued or running jobs"}
        rc, _, err = self.run(["scontrol", action, job_id])
        past = {"hold": "held", "release": "released", "requeue": "requeued"}[action]
        if rc != 0:
            return {action: False, "message": err.strip() or f"scontrol {action} failed"}
        return {action: True, "job_id": job_id, "message": f"Job {job_id} {past}"}

    READONLY_OK = {"squeue", "sinfo", "sacct", "sstat", "sprio", "sshare"}
    _SHELL_METACHARS = re.compile(r"[|;&><$`()\n]")

    def run_readonly_command(self, command):
        if self._SHELL_METACHARS.search(command):
            return {"error": "refused: shell metacharacters ( | ; & > < $ ` ( ) newline ) are not allowed"}
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return {"error": f"refused: cannot parse the command ({e})"}
        if argv[:2] == ["scontrol", "show"] or argv[:2] == ["sacctmgr", "show"]:
            ok = True
        elif argv and argv[0] in self.READONLY_OK:
            ok = True
        else:
            ok = False
        if not ok:
            return {"error": "refused: only read-only commands are allowed "
                             "(squeue, sinfo, sacct, sstat, sprio, sshare, scontrol show, sacctmgr show)"}
        rc, out, err = self.run(argv)
        if len(out) > 8000:
            out = out[:8000] + "[truncated]"
        return {"command": command, "exit_code": rc, "stdout": out, "stderr": err}

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
        # Finished jobs: sacct keeps the unexpanded --output/--error patterns. An empty StdErr means stderr went to
        # the StdOut file (no --error line); both empty means Slurm's default name in the working directory.
        fields = ["StdOut", "StdErr", "WorkDir", "JobName", "User", "JobIDRaw"]
        rc, out, _ = self.run(["sacct", "-n", "-P", "-X", "-j", job_id, f"--format={','.join(fields)}"])
        if rc == 0 and out.strip():
            row = dict(zip(fields, out.strip().splitlines()[0].split("|") + [""] * len(fields)))
            path = row[key] or row["StdOut"]
            if not path:
                return f"{row['WorkDir']}/slurm-{job_id}.out" if row["WorkDir"] else None
            path = _expand(path, job_id, row)
            return path if path.startswith("/") or not row["WorkDir"] else f"{row['WorkDir']}/{path}"
        return None


def _expand(pattern, job_id, row):
    """Slurm filename patterns (sbatch --output): %j job id, %A/%a array job id/index, %x job name, %u user, %% %."""
    array, _, index = job_id.partition("_")
    values = {"j": row.get("JobIDRaw") or array, "A": array, "a": index, "x": row.get("JobName", ""),
              "u": row.get("User", ""), "%": "%"}
    return re.sub(r"%(?:\d+)?([jAaxu%])", lambda m: values[m.group(1)], pattern)


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
    if name == "update_job":
        return f"set {args.get('field')}={args.get('value')} on job {args.get('job_id')} (scontrol update)"
    if name == "hold_job":
        return f"hold job {args.get('job_id')} so the scheduler does not start it (scontrol hold)"
    if name == "release_job":
        return f"release held job {args.get('job_id')} back to the scheduler (scontrol release)"
    if name == "requeue_job":
        return f"put job {args.get('job_id')} back in the queue to run again (scontrol requeue)"
    return f"{name} {json.dumps(args)}"
