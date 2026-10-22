"""Adversarial tests for unattended compute: evidence, crash recovery and stop conditions."""
import json
import subprocess
import sys
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/autopilot"))
import evidence as ev
import prepare as prep
import supervisor as sup
import configure as cfg
import submit_gpu as sg


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def comparison(tmp_path):
    code = tmp_path / "code"
    for label, path in ev.SLICES.items():
        jsonl(code / path, [{"id": label + str(i), "question": "question " + str(i)} for i in range(2)])
    out = code / "runs/vs_release_test"
    for label in ev.SLICES:
        for arm in ["base_closed", "rel12e_rag", *ev.ARMS]:
            jsonl(out / "answers" / f"{label}__{arm}.jsonl", [
                {"id": label + str(i), "answer": arm + str(i), "finish_reason": "stop"} for i in range(2)])
        for judge in ("judged", "judged_gemma"):
            for arm in ["rel12e_rag", *ev.ARMS]:
                jsonl(out / judge / f"{label}__{arm}__000.jsonl", [
                    {"id": label + str(i), "answer_arm": arm + str(i), "answer_base": "base_closed" + str(i),
                     "judge": {"arm_vs_base": ["T", "T"] if arm == "rel12e_rag" else ["A", "B"]}}
                    for i in range(2)])
    pairs = {f"{label}/{arm}-vs-rel12e_rag": {"questions": 2, "diff": .5, "p_a_not_better": .01}
             for label in ev.SLICES for arm in ev.ARMS}
    for name in ("PAIRED.json", "PAIRED_gemma.json"):
        put(out / name, pairs)
    return code, out


def test_goal_requires_both_judges_and_all_three_slices(comparison):
    code, out = comparison
    assert ev.promotion_decision(code, "test")["goal_met"]
    pairs = ev.read_json(out / "PAIRED_gemma.json")
    for arm in ev.ARMS:
        path = out / "judged_gemma" / f"d1__{arm}__000.jsonl"
        data = list(ev.rows(path).values())
        for row in data:
            row["judge"]["arm_vs_base"] = ["B", "A"]
        jsonl(path, data)
        pairs[f"d1/{arm}-vs-rel12e_rag"]["diff"] = -.5
    put(out / "PAIRED_gemma.json", pairs)
    decision = ev.promotion_decision(code, "test")
    assert not decision["goal_met"]
    assert all(not c["passed"] for c in decision["candidates"].values())


def test_missing_judgments_are_not_proved_by_summary(comparison):
    code, out = comparison
    path = out / "judged/v3__s42_rag__000.jsonl"
    jsonl(path, list(ev.rows(path).values())[:1])
    with pytest.raises(ev.EvidenceError, match="1/2 judgments"):
        ev.promotion_decision(code, "test")


def test_judgments_must_match_actual_answer_bytes(comparison):
    code, out = comparison
    path = out / "answers/v3__s42_rag.jsonl"
    data = list(ev.rows(path).values())
    data[0]["answer"] = "changed after judging"
    jsonl(path, data)
    with pytest.raises(ev.EvidenceError, match="binding mismatch"):
        ev.promotion_decision(code, "test")


def test_invalid_verdict_is_retained_and_disqualifies_arm(comparison):
    code, out = comparison
    path = out / "judged/v3__s42_rag__000.jsonl"
    data = list(ev.rows(path).values())
    data[0]["judge"]["arm_vs_base"] = [None, "B"]
    jsonl(path, data)
    pairs = ev.read_json(out / "PAIRED.json")
    pairs["v3/s42_rag-vs-rel12e_rag"]["questions"] = 1
    put(out / "PAIRED.json", pairs)
    decision = ev.promotion_decision(code, "test")
    assert not decision["candidates"]["s42_rag"]["passed"]
    assert decision["candidates"]["s42_rag"]["metrics"]["gpt_oss"]["v3"]["invalid"] == 1
    assert ev.rows(path)["v30"]["judge"]["arm_vs_base"][0] is None


def test_aggregate_numbers_checked_against_raw(comparison):
    code, out = comparison
    pairs = ev.read_json(out / "PAIRED.json")
    pairs["v3/s42_rag-vs-rel12e_rag"]["diff"] = .9
    put(out / "PAIRED.json", pairs)
    with pytest.raises(ev.EvidenceError, match="disagrees"):
        ev.promotion_decision(code, "test")


def test_significance_gate_and_nonfinite_results(comparison):
    code, out = comparison
    for name in ("PAIRED.json", "PAIRED_gemma.json"):
        pairs = ev.read_json(out / name)
        for v in pairs.values():
            v["p_a_not_better"] = .2
        put(out / name, pairs)
    assert not ev.promotion_decision(code, "test")["goal_met"]
    pairs = ev.read_json(out / "PAIRED.json")
    pairs["v3/s42_rag-vs-rel12e_rag"]["diff"] = float("nan")
    put(out / "PAIRED.json", pairs)
    assert not ev.promotion_decision(code, "test")["candidates"]["s42_rag"]["passed"]


def target_config(code, out):
    return {"alpha": .05, "target": {"minimum_score": .90, "rank": 1},
        "leaderboard_snapshot": {
            "baseline_answer_digests": {label: ev.answer_digest(ev.rows(out / "answers" / f"{label}__base_closed.jsonl"))
                                        for label in ev.SLICES},
            "judges": {judge: {"leader": {"mean_upper": .86}} for judge in ("gpt_oss", "gemma")}}}


def test_90_target_is_per_slice_not_mean_and_release_win_is_only_checkpoint(comparison):
    code, out = comparison
    config = target_config(code, out)
    assert ev.objective_decision(code, "test", config)["goal_met"]
    pairs = ev.read_json(out / "PAIRED_gemma.json")
    for arm in ev.ARMS:
        path = out / "judged_gemma" / f"d1__{arm}__000.jsonl"
        records = list(ev.rows(path).values())
        records[0]["judge"]["arm_vs_base"] = ["T", "B"]
        jsonl(path, records)
        pairs[f"d1/{arm}-vs-rel12e_rag"]["diff"] = .375
    put(out / "PAIRED_gemma.json", pairs)
    decision = ev.objective_decision(code, "test", config)
    assert decision["release_checkpoint_met"]
    assert not decision["goal_met"]
    assert decision["leaderboard"]["s42_rag"]["judges"]["gemma"]["mean"] > .90


def test_90_target_also_requires_first_place_and_identical_reference(comparison):
    code, out = comparison
    config = target_config(code, out)
    config["leaderboard_snapshot"]["judges"]["gemma"]["leader"]["mean_upper"] = 1
    assert not ev.objective_decision(code, "test", config)["goal_met"]
    config["leaderboard_snapshot"]["baseline_answer_digests"]["v3"] = "wrong"
    with pytest.raises(ev.EvidenceError, match="same baseline answers"):
        ev.objective_decision(code, "test", config)


def test_duplicate_answer_ids_fail_even_when_line_count_matches(tmp_path):
    path = tmp_path / "answers.jsonl"
    jsonl(path, [{"id": "a", "answer": "one", "finish_reason": "stop"}] * 2)
    with pytest.raises(ev.EvidenceError, match="Duplicate"):
        ev.answer_rows(path, {"a", "b"})


def test_factory_release_is_hash_bound(tmp_path):
    path = tmp_path / "runs/batch/release"
    jsonl(path / "verified.jsonl", [{"id": "a"}])
    put(path / "REPORT.json", {"status": "automated_gates_passed_not_human_reviewed", "released_rows": 1,
                                "sha256": ev.sha256(path / "verified.jsonl")})
    assert ev.audit_factory(tmp_path, "batch")["released_rows"] == 1
    jsonl(path / "verified.jsonl", [{"id": "changed"}])
    with pytest.raises(ev.EvidenceError, match="digest"):
        ev.audit_factory(tmp_path, "batch")


@pytest.fixture
def service(tmp_path):
    config = tmp_path / "config.json"
    put(config, {"root": str(tmp_path), "code": str(tmp_path / "code"), "python": sys.executable,
                 "account": "test", "cpu_partition": "test", "pinned_files": {}, "alpha": .05})
    return sup.Supervisor(config)


def test_scheduler_failure_is_not_empty_queue(service, monkeypatch):
    monkeypatch.setattr(service.slurm, "command", lambda *a, **kw: (_ for _ in ()).throw(sup.SchedulerError("offline")))
    with pytest.raises(sup.SchedulerError):
        service.slurm.queue()


def test_duplicate_coordinators_cannot_take_lock(tmp_path):
    path = tmp_path / "coordinator.lock"
    with sup.lock(path) as acquired:
        assert acquired
        with sup.lock(path) as second:
            assert not second


def test_crash_after_submission_recovers_without_duplicate(service, monkeypatch):
    path = service.directory / "submissions/key.json"
    sup.atomic(path, {"name": "dka-unique", "since": "2026-10-22T00:00:00", "state": "intent"})
    monkeypatch.setattr(service.slurm, "queue", lambda: [{"id": "123", "name": "dka-unique", "state": "RUNNING"}])
    monkeypatch.setattr(service.slurm, "command", lambda *a, **kw: "123|dka-unique|RUNNING\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not resubmit"))
    assert service.slurm.submit("key", Path("unused")) == "123"
    assert ev.read_json(path)["job_id"] == "123"


def test_ambiguous_submission_waits_not_resubmits(service, monkeypatch):
    path = service.directory / "submissions/key.json"
    sup.atomic(path, {"name": "dka-unique", "since": "2026-10-22T00:00:00", "state": "intent"})
    monkeypatch.setattr(service.slurm, "queue", lambda: [])
    monkeypatch.setattr(service.slurm, "command", lambda *a, **kw: "")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not resubmit"))
    with pytest.raises(sup.SchedulerError, match="Unresolved"):
        service.slurm.submit("key", Path("unused"))


def test_training_submitter_adopts_live_job_and_respects_cancellation(service, monkeypatch):
    monkeypatch.setattr(sup.Slurm, "submit", lambda *a, **kw: "123")
    assert sg.submit(str(service.config_path), ["--nodes=8", "script"]) == "123"
    monkeypatch.setattr(sup.Slurm, "submit", lambda *a, **kw: pytest.fail("must not duplicate"))
    monkeypatch.setattr(sup.Slurm, "status", lambda *a: ("RUNNING", "0:0"))
    assert sg.submit(str(service.config_path), ["--nodes=8", "script"]) == "123"
    monkeypatch.setattr(sup.Slurm, "status", lambda *a: ("CANCELLED", "0:0"))
    with pytest.raises(ev.EvidenceError, match="cancelled"):
        sg.submit(str(service.config_path), ["--nodes=8", "script"])


def test_done_log_does_not_advance_without_artifacts(service, monkeypatch):
    log = service.directory / "old.log"
    log.parent.mkdir(parents=True)
    log.write_text("DONE: Traceback: aggregation failed\n")
    monkeypatch.setattr(service.slurm, "submit", lambda *a, **kw: "101")
    def missing():
        raise ev.EvidenceError("result missing")
    assert not service.task("aggregate", ["true"], missing)
    assert service.state["tasks"]["aggregate"]["status"] == "running"


def test_repeated_real_failures_stop_but_timeout_resumes(service, monkeypatch):
    monkeypatch.setattr(service.slurm, "submit", lambda *a, **kw: "101")
    monkeypatch.setattr(service.slurm, "status", lambda *a: ("TIMEOUT", "0:0"))
    def missing():
        raise ev.EvidenceError("pending")
    for _ in range(4):
        assert not service.task("round", ["true"], missing)
    assert service.state["tasks"]["round"]["failures"] == 0
    monkeypatch.setattr(service.slurm, "status", lambda *a: ("FAILED", "1:0"))
    for _ in range(2):
        assert not service.task("round", ["true"], missing)
    with pytest.raises(ev.EvidenceError, match="3 failed attempts"):
        service.task("round", ["true"], missing)


def test_manual_cancellation_is_respected(service, monkeypatch):
    service.state["tasks"]["job"] = {"attempt": 1, "failures": 0, "job_id": "9", "status": "running"}
    monkeypatch.setattr(service.slurm, "status", lambda *a: ("CANCELLED", "0:0"))
    def missing():
        raise ev.EvidenceError("missing")
    with pytest.raises(ev.EvidenceError, match="cancelled"):
        service.task("job", ["true"], missing)


def test_pause_and_stop_do_not_submit(service, monkeypatch):
    monkeypatch.setattr(service.slurm, "queue", lambda: [])
    service.directory.mkdir(parents=True)
    (service.directory / "PAUSE").touch()
    service.tick()
    assert service.state["status"] == "paused"
    (service.directory / "STOP").touch()
    service.tick()
    assert service.state["status"] == "stopped"


def test_successor_chain_survives_one_failed_cluster_start(service, monkeypatch):
    service.config['python'] = sys.executable
    receipts, submissions = {}, []
    def submit(key, script, *, dependency):
        if key not in receipts:
            receipts[key] = str(101 + len(receipts))
            submissions.append((key, dependency))
        return receipts[key]
    monkeypatch.setattr(service.slurm, 'submit', submit)
    service.directory.mkdir(parents=True)
    assert service.successors('100') == ['101', '102', '103']
    assert submissions == [('service-after-100', '100'), ('service-after-101', '101'), ('service-after-102', '102')]
    # 101 fails before Python starts; 102 already exists and only replenishes the tail.
    assert service.successors('102') == ['103', '104', '105']
    assert len(submissions) == 5
    assert service.state['service']['standby_job_ids'] == ['104', '105']


def test_initial_service_can_be_held_until_its_recovery_chain_exists(service, monkeypatch):
    script = service.config_path.parent / 'service.sh'
    script.write_text('#!/bin/bash\ntrue\n')
    monkeypatch.setattr(service.slurm, 'queue', lambda: [])
    def submit(args, **kwargs):
        assert '--hold' in args
        assert '--mem=2G' in args
        return subprocess.CompletedProcess(args, 0, stdout='123\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', submit)
    assert service.slurm.submit('service-held', script, hold=True) == '123'


def test_nested_controller_does_not_inherit_slurm_allocation(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    put(spec, {"command": ["true"], "cwd": str(tmp_path), "receipt": str(tmp_path / "result.json")})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1024")
    def run(command, *, cwd, env):
        assert not any(k.startswith("SLURM_") for k in env)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", run)
    assert sup.run_task(spec) == 0


def test_preparation_is_repeatable_and_does_not_lower_threshold(tmp_path, monkeypatch):
    code = tmp_path / "code"
    bank = code / "teacher.jsonl"
    jsonl(bank, [{"id": f"row{i}", "messages": [
        {"role": "user", "content": f"Provided input 1 (quoted data): unique case {i}"},
        {"role": "assistant", "content": "source grounded answer"}]} for i in range(1800)])
    for path in ev.SLICES.values():
        jsonl(code / path, [{"id": "eval", "question": "never train on this held-out question"}])
    config = {"code": str(code), "root": str(tmp_path), "teacher_bank": "teacher.jsonl",
              "teacher_bank_sha256": ev.sha256(bank), "baseline_model": "release"}
    a = prep.prepare(config, 1)
    assert a == prep.prepare(config, 1)
    b = prep.prepare(config, 2)
    assert a["sample_seed"] != b["sample_seed"]
    assert set(ev.rows(code / "runs/auto_rft_0001/data/questions.jsonl")).isdisjoint(
        ev.rows(code / "runs/auto_rft_0002/data/questions.jsonl"))
    script = (tmp_path / "ctl/auto_rft_0001/controller.sh").read_text()
    assert "keeping ties" not in script
    assert "want * 98" not in script
    assert "STOP: aggregation failed" in script
    assert "autopilot_wait" in script
    assert "submit_gpu.py" in script and "sbatch --parsable" not in script
    assert "wait_gpu.py" in script and 'squeue -h -j "$1"' not in script
    assert "--nodes=8 --job-name=dk-auto_rft_0001-gen" in script
    assert a["training_prompt_policy"] == "concise_completion"
    assert b["training_prompt_policy"] == "matched_prompt"


def test_private_controller_rendering(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    shutil.copytree(ROOT / "research/autopilot/templates", runtime / "templates")
    monkeypatch.setattr(cfg, "__file__", str(runtime / "configure.py"))
    monkeypatch.setattr(cfg, "leaderboard_snapshot", lambda code: {"arms": ["fixture"]})
    code = tmp_path / "code"
    for source in ("research/bestofn/candidates.py", "research/kd/onpolicy_gen.sbatch", "research/rag65/tool_eval.sbatch", "research/gguf/export.sh"):
        dest = code / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, dest)
    for path in ev.SLICES.values():
        jsonl(code / path, [{"id": "test", "question": "held out"}])
    jsonl(code / "data/eval/dataset_v1/development_tools.jsonl", [{"id": "tool"}])
    for name in ("rag65_seed42", "rag65kd_seed42", "rag65kd_seed43", "rag65kd_seed44"):
        put(code / "configs" / (name + ".json"), {})
    jsonl(code / "runs/rag65v9/data/verified.jsonl", [{"id": "training"}])
    weights = code / "runs/gguf_r12e/merged/model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture weights")
    config = cfg.render(tmp_path, sys.executable)
    stage = (runtime / "stage_release.sh").read_text()
    assert "approved-arm" in stage and "best = None" not in stage
    assert "audit-export" in stage
    assert "model*.safetensors" in stage
    assert "srun -A" not in stage
    assert "--nodes=8" in (runtime / "vs_release.sh").read_text()
    assert "MERGED_READY" in stage
    assert "model-$q.partial.gguf" in (runtime / "export_gguf.sh").read_text()
    compile((runtime / "candidates.py").read_text(), "candidates.py", "exec")
    assert "int(os.environ['AUTOPILOT_SAMPLE_SEED']) + k" in (runtime / "candidates.py").read_text()
    assert 'start_vllm "${BASE_MODEL:?}"' in (runtime / "tool_eval.sbatch").read_text()
    assert config["baseline_weights"]
    with pytest.raises(ev.EvidenceError, match="already exists"):
        cfg.render(tmp_path, sys.executable)
