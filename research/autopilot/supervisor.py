"""Persistent Slurm coordinator. No credentials, model calls or GPU work on the coordinator."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

if __name__ == '__main__':
    sys.modules.setdefault('supervisor', sys.modules[__name__])

from evidence import (ARMS, EvidenceError, answer_rows, audit_factory, audit_judgments,
    audit_round, expected_ids, objective_decision, promotion_decision, read_json, sha256)


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "REQUEUE_FED", "RESIZING"}
RETRYABLE = {"TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
TERMINAL = {"goal_met", "blocked", "stopped"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


@contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class SchedulerError(RuntimeError):
    pass


class Slurm:
    def __init__(self, config, directory):
        self.config, self.directory = config, Path(directory)

    def command(self, args, timeout=45):
        env = {**os.environ, "TZ": "UTC"}
        try:
            p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SchedulerError(f"Scheduler unavailable: {args[0]}: {exc}") from exc
        if p.returncode:
            raise SchedulerError(f"{args[0]} exited {p.returncode}: {p.stderr.strip()[:600]}")
        return p.stdout

    def queue(self):
        # Never convert a scheduler failure into an empty queue.
        output = self.command(["squeue", "--me", "-h", "-o", "%i|%100j|%T"])
        result = []
        for line in output.splitlines():
            parts = [s.strip() for s in line.split("|")]
            if len(parts) != 3 or not re.fullmatch(r"\d+(?:[_+].+)?", parts[0]):
                raise SchedulerError(f"Unexpected squeue response: {line[:120]}")
            result.append({"id": parts[0], "name": parts[1], "state": parts[2]})
        return result

    def status(self, job):
        output = self.command(["sacct", "-n", "-X", "-P", "-j", str(job),
                               "--format=JobIDRaw,State,ExitCode"])
        for line in output.splitlines():
            parts = line.split("|")
            if len(parts) >= 3 and parts[0] == str(job):
                return parts[1].split()[0].rstrip("+"), parts[2]
        # Accounting can lag; absence is not proof of termination.
        return "UNKNOWN", None

    def recover_intent(self, receipt):
        matches = {j["id"] for j in self.queue() if j["name"] == receipt["name"]}
        output = self.command(["sacct", "-n", "-X", "-P", "-S", receipt["since"],
                               "--format=JobIDRaw,JobName%100,State"])
        for line in output.splitlines():
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 3 and p[0].isdigit() and p[1] == receipt["name"]:
                matches.add(p[0])
        if len(matches) > 1:
            raise EvidenceError("More than one job matches a unique submission receipt")
        return next(iter(matches), None)

    def submit(self, key, script, *, dependency=None, controller=False, gpu_args=None, environment=None):
        path = self.directory / "submissions" / f"{key}.json"
        if path.exists():
            receipt = read_json(path)
            if receipt.get("job_id"):
                return receipt["job_id"]
            recovered = self.recover_intent(receipt)
            if recovered:
                receipt.update(job_id=recovered, recovered_at=now())
                atomic(path, receipt)
                return recovered
            if receipt.get("state") != "refused":
                raise SchedulerError(f"Unresolved submission {key}; waiting for scheduler evidence, not resubmitting")
        # Check scheduler health BEFORE persisting a new submission intent.
        self.queue()
        name = "dka-" + hashlib.sha256(key.encode()).hexdigest()[:20]
        receipt = {"name": name, "key": key, "since": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                   "script": str(script), "script_sha256": sha256(script), "state": "intent", "at": now()}
        atomic(path, receipt)
        log = self.directory / "logs" / f"{key}-%j"
        log.parent.mkdir(parents=True, exist_ok=True)
        args = ["sbatch", "--parsable", "--account=" + self.config["account"],
            "--partition=" + self.config["cpu_partition"], "--qos=normal", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=" + ("4" if controller else "1"), "--mem=" + ("28G" if controller else "2G"),
            "--time=04:00:00", "--job-name=" + name, "--comment=distillkit-autopilot:" + key,
            "--output=" + str(log) + ".out", "--error=" + str(log) + ".err"]
        if dependency:
            args.append("--dependency=afterany:" + str(dependency))
        args.append(str(script))
        if gpu_args is not None:
            args = ["sbatch", "--parsable", *gpu_args[:-1], "--job-name=" + name,
                    "--comment=distillkit-autopilot:" + key,
                    "--output=" + str(log) + ".out", "--error=" + str(log) + ".err",
                    "--account=" + self.config["account"], "--time=00:30:00", "--qos=boost_qos_dbg",
                    "--partition=boost_usr_prod", gpu_args[-1]]
        try:
            # An interrupted response is ambiguous. The durable intent is reconciled on the next tick.
            env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_'))}
            env.update(environment or {})
            p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SchedulerError(f"Submission response unknown for {key}: {exc}") from exc
        job = p.stdout.strip().split(";")[0]
        if p.returncode == 0 and job.isdigit():
            receipt.update(job_id=job, state="submitted", submitted_at=now())
            atomic(path, receipt)
            return job
        # Only a definite sbatch rejection can be retried; malformed success remains ambiguous.
        receipt.update(state="refused" if p.returncode else "unknown", error=(p.stderr + p.stdout)[-1200:])
        atomic(path, receipt)
        raise SchedulerError(f"Submission not confirmed for {key}: {receipt['error']}")


class Supervisor:
    def __init__(self, config_path, scheduler=None):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(self.config_path)
        self.directory = Path(self.config["root"]) / "ctl/autopilot"
        self.runtime = Path(__file__).resolve().parent
        self.code = Path(self.config["code"])
        self.path = self.directory / "STATE.json"
        self.state = read_json(self.path) if self.path.exists() else {
            "version": 1, "status": "running", "phase": "data", "round_index": 1,
            "history_index": 0, "tasks": {}, "decisions": [], "created_at": now()}
        self.slurm = scheduler or Slurm(self.config, self.directory)

    def save(self):
        self.state["updated_at"] = now()
        atomic(self.path, self.state)
        atomic(self.directory / "STATUS.json", {k: v for k, v in self.state.items() if k != "tasks"})
        active = [(k, v) for k, v in self.state["tasks"].items() if v.get("status") == "running"]
        lines = ["# HPC pipeline automation", "", f"Updated: {self.state['updated_at']}",
            f"Status: **{self.state['status']}**; phase: **{self.state['phase']}**.", "",
            "Target: at least 90% preference score on each of v2, v3 and d1 under both judges, plus first place on the frozen internal leaderboard. Beating release 12e is a checkpoint.",
            "", "These are development comparisons. The final holdout remains untouched.", "",
            "GPU jobs retain the 30-minute debug limit. The coordinator and controller jobs use the CPU partition.", ""]
        if self.state.get("data"):
            lines.append(f"Latest data batch: {self.state['data']['released_rows']:,} rows passed automated gates; not human reviewed.")
        for task, record in active:
            lines.append(f"Active task: `{task}`; CPU controller job `{record.get('job_id')}`; attempt {record.get('attempt')}.")
        if self.state.get("current_run"):
            lines.append(f"Current candidate: `{self.state['current_run']}`.")
        if self.state.get('gpu_pool'):
            pool = self.state['gpu_pool']
            lines.append(f"GPU pool: {pool.get('running_nodes', '?')}/8 nodes running; {pool.get('running_gpus', '?')}/32 GPUs allocated; {pool.get('reserved_nodes', '?')} nodes reserved by queued/running jobs.")
        if self.state.get('backfill'):
            lines.append('Background preparation: ' + self.state['backfill'].get('detail', 'starting'))
        for decision in self.state["decisions"][-5:]:
            lines.append(f"Candidate `{decision['run']}`: {'met' if decision['goal_met'] else 'did not meet'} the 90% and first-place target under both judges.")
            if decision.get("best_observed"):
                best = decision["best_observed"]
                lines.append(f"Best minimum slice score: {best['minimum_score']:.1%} (`{best['arm']}`); means and conservative ranks: {best['judges']}.")
        if self.state.get("detail"):
            lines.extend(["", self.state["detail"]])
        lines.extend(["", "Controls: create ctl/autopilot/PAUSE to stop new submissions; remove it to resume. Create STOP to stop the coordinator. Existing GPU jobs are not cancelled.",
            "", "Publication is separate. A successful candidate is exported, checked and staged with a model card and evidence; the currently published model is not overwritten by this service."])
        path = self.directory / "STATUS.md"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(path)

    def check_pins(self):
        for relative, digest in self.config.get("pinned_files", {}).items():
            if sha256(self.code / relative) != digest:
                raise EvidenceError(f"Frozen code/evaluation input changed: {relative}")

    def task(self, key, command, proof):
        record = self.state["tasks"].setdefault(key, {"attempt": 0, "failures": 0})
        command_hash = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        if record.get("command_hash", command_hash) != command_hash:
            raise EvidenceError(f"Task command changed after preparation: {key}")
        record["command_hash"] = command_hash
        if record.get("job_id") and record.get("status") != "complete":
            state, exit_code = self.slurm.status(record["job_id"])
            record["scheduler_state"] = state
            if state in ACTIVE or state == "UNKNOWN":
                return False
        else:
            state, exit_code = None, None
        try:
            evidence = proof()
        except (EvidenceError, OSError) as exc:
            record["evidence_pending"] = str(exc)
        else:
            record.update(status="complete", evidence=evidence, completed_at=now())
            return True
        if record.get("job_id") and state not in ACTIVE and state != "UNKNOWN":
            if state == "CANCELLED":
                raise EvidenceError(f"{key} was cancelled; respecting the cancellation instead of restarting it")
            if state not in RETRYABLE:
                record["failures"] += 1
            record["previous_job_id"] = record.pop("job_id")
            record["status"] = "retrying"
            self.save()
        if record["failures"] >= 3 or record["attempt"] >= 96:
            raise EvidenceError(f"{key} stopped after {record['failures']} failed attempts: {record['evidence_pending']}")
        attempt = record["attempt"] + 1
        submission = f"task-{key}-{attempt}"
        script = self.directory / "jobs" / f"{submission}.sh"
        task_spec = self.directory / "jobs" / f"{submission}.json"
        spec = {"command": command, "cwd": str(self.code), "receipt": str(task_spec.with_suffix('.result.json'))}
        atomic(task_spec, spec)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/bash\nexec " + shlex.join([self.config["python"], str(self.runtime / "supervisor.py"),
            "run-task", str(task_spec)]) + "\n")
        record["job_id"] = self.slurm.submit(submission, script, controller=True)
        record.update(attempt=attempt, status="running", submitted_at=now())
        self.save()
        return False

    def transition(self, phase):
        self.state.update(phase=phase, detail="")
        self.save()

    def compare(self, run):
        script = str(self.runtime / "vs_release.sh")
        return self.task("compare-" + run, ["bash", script, run],
                         lambda: promotion_decision(self.code, run, alpha=self.config["alpha"]))

    def teacher(self, run):
        def proof():
            directory = self.code / "runs" / f"vs_teacher_{run}"
            read_json(directory / "RESULT.json")
            data = audit_judgments(self.code, directory, "judged", ARMS)
            return {"invalid": sum(r["invalid"] for r in data.values()), "judged_arms": len(data)}
        return self.task("teacher-" + run, ["bash", str(self.runtime / "vs_teacher.sh"), run], proof)

    def decision(self, run):
        decision = objective_decision(self.code, run, self.config)
        atomic(self.directory / "decisions" / (run + ".json"), decision)
        if not any(x["run"] == run for x in self.state["decisions"]):
            record = {"run": run, "goal_met": decision["goal_met"],
                "selected_arm": decision["selected_arm"], "at": now()}
            standings = decision.get("leaderboard", {})
            candidates = {a: item for a, item in standings.items() if len(item["judges"]) == 2}
            if candidates:
                minimum = lambda a: min(s for j in candidates[a]["judges"].values() for s in j["scores"].values())
                arm = max(candidates, key=minimum)
                record["best_observed"] = {"arm": arm, "minimum_score": minimum(arm),
                    "judges": {j: {"mean": m["mean"], "rank": m["rank_conservative"]}
                               for j, m in candidates[arm]["judges"].items()}}
            self.state["decisions"].append(record)
        if decision["goal_met"]:
            self.state["current_run"] = run
            self.state["winner"] = decision
            self.transition("candidate_checks")
            return True
        return False

    def tick(self):
        self.check_pins()
        self.slurm.queue()
        if (self.directory / "STOP").exists():
            self.state.update(status="stopped", detail="STOP file is present; existing jobs were left alone.")
            return
        if (self.directory / "PAUSE").exists():
            self.state.update(status="paused", detail="PAUSE is present; no new submissions.")
            return
        if self.state["status"] in TERMINAL:
            return
        self.state.update(status="running", detail="")
        phase = self.state["phase"]
        if self.config.get('pool', {}).get('enabled') and phase in ('prepare', 'train_evaluate'):
            run = f"auto_rft_{self.state['round_index']:04d}"
            rejected = self.code / 'runs' / run / 'DATA_REJECTED.json'
            if rejected.exists():
                record = self.state['tasks'].get('round-' + run, {})
                if record.get('job_id'):
                    state, _ = self.slurm.status(record['job_id'])
                    if state in ACTIVE or state == 'UNKNOWN':
                        return
                    record['status'] = 'rejected'
                self.state['decisions'].append({'run': run, 'goal_met': False, 'selected_arm': None,
                    'reason': read_json(rejected)['reason'], 'at': now()})
                self.state['round_index'] += 1
                self.transition('prepare')
                return
        if phase == "data":
            factory = self.config["factory"]
            if self.task("factory", ["bash", str(Path(self.config["root"]) / "ctl" / factory["controller"] / "controller.sh")],
                         lambda: audit_factory(self.code, factory["batch"])):
                self.state["data"] = self.state["tasks"]["factory"]["evidence"]
                self.transition("history")
        elif phase == "history":
            history = self.config["existing_rounds"]
            i = self.state["history_index"]
            if i >= len(history):
                self.transition("prepare")
            else:
                run = history[i]
                audit_round(self.code, run)
                if self.teacher(run) and self.compare(run):
                    if not self.decision(run):
                        self.state["history_index"] += 1
        elif phase == "prepare":
            index = self.state["round_index"]
            run = f"auto_rft_{index:04d}"
            self.state["current_run"] = run
            def proof():
                plan = read_json(self.code / "runs" / run / "AUTOPILOT_PLAN.json")
                if sha256(self.code / "runs" / run / "data/questions.jsonl") != plan["question_sha256"]:
                    raise EvidenceError("Prepared question hash changed")
                return plan
            if self.config.get('pool', {}).get('enabled'):
                # The coordinator also owns backfill preparation; never race a second CPU writer.
                if not (self.code / 'runs' / run / 'AUTOPILOT_PLAN.json').exists():
                    from prepare import prepare
                    prepare(self.config, index)
                proof()
                self.transition('train_evaluate')
            elif self.task("prepare-" + run, [self.config["python"], str(self.runtime / "prepare.py"), str(self.config_path), str(index)], proof):
                self.transition("train_evaluate")
        elif phase == "train_evaluate":
            run = self.state["current_run"]
            if self.task("round-" + run, ["bash", str(Path(self.config["root"]) / "ctl" / run / "controller.sh")],
                         lambda: audit_round(self.code, run)):
                self.transition("compare")
        elif phase == "compare":
            run = self.state["current_run"]
            if self.teacher(run) and self.compare(run):
                if not self.decision(run):
                    self.state["round_index"] += 1
                    self.transition("prepare")
        elif phase == "candidate_checks":
            run = self.state["current_run"]
            if self.task("checks-" + run, [self.config["python"], str(self.runtime / "candidate_checks.py"), str(self.config_path), run],
                         lambda: read_json(self.code / "runs" / run / "AUTOPILOT_CHECKS.json")):
                checks = read_json(self.code / "runs" / run / "AUTOPILOT_CHECKS.json")
                if checks["passed"]:
                    self.transition("export")
                else:
                    self.state["detail"] = "Benchmark winner failed tool/command non-regression checks; continuing research."
                    self.state["round_index"] += 1
                    self.transition("prepare")
        elif phase == "export":
            run = self.state["current_run"]
            if self.task("export-" + run, ["bash", str(self.runtime / "stage_release.sh"), run], lambda: self.export_proof(run)):
                self.state.update(status="goal_met", detail=f"{run} reached 90% on all three development benchmarks and first place under both judges; model and GGUF artifacts staged in runs/hf_{run}.")
                atomic(self.directory / "GOAL_MET.json", {"at": now(), **self.state["winner"]})
        else:
            raise EvidenceError(f"Unknown phase: {phase}")
        if self.config.get('pool', {}).get('enabled') and self.state['status'] not in TERMINAL and self.state.get('data'):
            from backfill import Backfill
            Backfill(self).tick()

    def export_proof(self, run, require_ready=True):
        directory = self.code / "runs" / f"hf_{run}"
        if require_ready and not (directory / "READY").is_file():
            raise EvidenceError("Release READY file missing")
        weights = sorted(p.name for p in directory.glob("model*.safetensors"))
        if not weights:
            raise EvidenceError("No staged model weights")
        index = directory / "model.safetensors.index.json"
        if index.exists():
            if not set(read_json(index)["weight_map"].values()) <= set(weights):
                raise EvidenceError("Staged model is missing a weight shard")
        files = [*weights, "config.json", "tokenizer.json", "README.md",
                 "gguf/hpc-assistant-qwen3-4b-Q4_K_M.gguf", "gguf/hpc-assistant-qwen3-4b-Q8_0.gguf"]
        for name in files:
            p = directory / name
            if not p.is_file() or not p.stat().st_size:
                raise EvidenceError(f"Release artifact missing/empty: {name}")
        evaluation = self.code / "runs" / f"gguf_{run}_eval"
        for arm in ("cand_gguf_q80_rag", "cand_gguf_q4kmimat_rag", "cand_vllm_rag"):
            for label, ids in expected_ids(self.code).items():
                answer_rows(evaluation / "answers" / f"{label}__{arm}.jsonl", ids)
        judged = audit_judgments(self.code, evaluation, "judged",
                                ["cand_gguf_q80_rag", "cand_gguf_q4kmimat_rag", "cand_vllm_rag"])
        if any(r["invalid"] for r in judged.values()):
            raise EvidenceError("Quantized evaluation contains invalid judgments; release remains staged")
        read_json(evaluation / "RESULT.json")
        return {"directory": str(directory), "files": files, "quantization_comparison": str(evaluation / "PAIRED.json")}

    def service_script(self):
        p = self.directory / "service.sh"
        p.write_text("#!/bin/bash\nexec " + shlex.join([self.config["python"], str(self.runtime / "supervisor.py"),
            "serve", str(self.config_path)]) + "\n")
        return p

    def successors(self, current_job):
        # Queue beyond the immediate successor: it may fail in the cluster prolog before Python
        # can queue its own replacement. All links use durable, shared submission keys.
        chain, parent = [], str(current_job)
        script = self.service_script()
        for _ in range(self.config.get('service_successor_depth', 3)):
            parent = self.slurm.submit('service-after-' + parent, script, dependency=parent)
            chain.append(parent)
        self.state['service'] = {'job_id': str(current_job), 'successor_job_id': chain[0], 'standby_job_ids': chain[1:]}
        return chain

    def serve(self, once=False):
        with lock(self.directory / "coordinator.lock") as acquired:
            if not acquired:
                print("Another coordinator owns the lock.")
                return
            if self.state["status"] in TERMINAL:
                self.save()
                return
            current_job = os.environ.get("SLURM_JOB_ID")
            if not once and not current_job:
                raise EvidenceError("Use bootstrap to start the durable Slurm service")
            deadline = time.monotonic() + self.config.get("service_seconds", 13800)
            while True:
                try:
                    # Prequeue a successor: it can recover even when this allocation/node dies.
                    if not once and self.state["status"] not in TERMINAL:
                        self.successors(current_job)
                    self.tick()
                except SchedulerError as exc:
                    self.state.update(detail=str(exc), scheduler_error_at=now())
                except (EvidenceError, OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
                    self.state.update(status="blocked", detail=f"Automation stopped for inspection: {exc}")
                self.save()
                if self.config.get('reports', {}).get('enabled'):
                    try:
                        from reports import write_report
                        write_report(self.config, self.state, self.slurm)
                    except (SchedulerError, EvidenceError, OSError, KeyError, TypeError, ValueError) as exc:
                        # A reporting outage must not stop successful GPU work.
                        atomic(self.directory / 'REPORT_ERROR.json', {'at': now(), 'error': str(exc)})
                if once or self.state["status"] in TERMINAL or time.monotonic() >= deadline:
                    return
                time.sleep(self.config.get("poll_seconds", 60))


def run_task(path):
    spec = read_json(path)
    # Controllers submit independent allocations. They must not mistake the CPU controller's allocation
    # for the target of a nested srun/sbatch, or export its resource settings to GPU jobs.
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    started = now()
    p = subprocess.run(spec["command"], cwd=spec["cwd"], env=env)
    atomic(spec["receipt"], {"started_at": started, "finished_at": now(), "returncode": p.returncode,
                             "command": spec["command"]})
    return p.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bootstrap", "serve", "once", "status", "run-task", "audit-export"])
    parser.add_argument("config")
    parser.add_argument("run", nargs="?")
    args = parser.parse_args()
    if args.action == "run-task":
        return run_task(args.config)
    s = Supervisor(args.config)
    if args.action == "bootstrap":
        with lock(s.directory / "bootstrap.lock") as acquired:
            if not acquired:
                return 0
            jobs = [j for j in s.slurm.queue() if j["name"].startswith("dka-")]
            if jobs:
                print(json.dumps({"already_managed_jobs": jobs}))
            elif s.state["status"] not in TERMINAL:
                print(json.dumps({"service_job_id": s.slurm.submit("service-initial", s.service_script())}))
    elif args.action == "status":
        print(json.dumps(s.state, indent=2))
    elif args.action == "audit-export":
        print(json.dumps(s.export_proof(args.run, require_ready=False), indent=2))
    else:
        s.serve(once=args.action == "once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
