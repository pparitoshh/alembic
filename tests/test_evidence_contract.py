"""Real generation/verification with fake completions and isolated local tool fixtures. No network."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from types import SimpleNamespace

import pytest

from distillkit import generate, toolcheck, tools, verify
from distillkit.io import read_jsonl, write_jsonl
from distillkit.teacher import Completion
from test_tools import ROOT, _cfg, _call


def test_source_tool_selection_is_explicit_deterministic_and_preserves_holdout(tmp_path):
    cfg = _cfg(tmp_path, tool_fraction=1, tool_doc_ids=["slurm_monitoring"])
    jobs = generate.question_jobs(cfg)
    assert jobs == generate.question_jobs(cfg)
    assert len({j["id"] for j in jobs}) == len(jobs)
    assert {j["doc_id"] for j in jobs if j["mode"] != "prose"} == {"slurm_monitoring"}
    assert not set(cfg.seeds.eval_docs) & {j["doc_id"] for j in jobs}
    assert len(jobs) == len(generate.question_jobs(_cfg(tmp_path, tool_fraction=0)))


@pytest.mark.parametrize("doc", ["missing", "slurm_job_arrays", "slurm_requeue_signals"])
def test_unresolved_tool_source_stops_before_teacher(tmp_path, monkeypatch, doc):
    cfg = _cfg(tmp_path, tool_doc_ids=[doc])
    monkeypatch.setattr(generate, "Teacher", lambda *_: pytest.fail("teacher constructed before source validation"))
    with pytest.raises(ValueError, match="forbidden or unknown"):
        generate.run(cfg)
    assert not cfg.run_dir.exists()


def test_actual_question_prompt_and_record_receive_complete_scenario(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, questions_per_chunk=1, tool_fraction=1, tool_mix={"call": 1}, gold_dir=None)
    captured = []

    class Teacher:
        model = "synthetic"
        def __init__(self, *_): pass
        def map(self, fn, items): return [fn(i) for i in items]
        def chat_json(self, system, prompt, schema):
            captured.append(prompt)
            return SimpleNamespace(question="Synthetic resource inspection."), ""
        def complete(self, *args, **kwargs): return Completion(content="Synthetic answer.")

    monkeypatch.setattr(generate, "Teacher", Teacher)
    generate.run(cfg)
    rows = read_jsonl(cfg.run_dir / "questions.jsonl")
    assert len(rows) == len(captured) > 0
    for row, prompt in zip(rows, captured):
        job = row["scenario"]["job"]
        assert job == tools.execute("job_status", {"job_id": job["job_id"]})
        assert "gpus" in job and "time_limit" in job
        assert json.dumps(row["scenario"], sort_keys=True) in prompt
        assert row["mock_version"] == tools.MOCK_VERSION


def test_logs_and_accounting_use_the_canonical_job():
    for job_id in ("4941833", "4229192", "4718209"):
        job = tools.execute("job_status", {"job_id": job_id})
        stdout = tools.execute("read_job_log", {"job_id": job_id, "stream": "stdout"})["lines"]
        if job["gpus"]:
            assert stdout[0].startswith(f"Using {job['gpus']} GPUs:")
        else:
            assert stdout[0] == "CPU-only job; no GPUs allocated"
        stderr = tools.execute("read_job_log", {"job_id": job_id})["lines"]
        if job["state"] == "FAILED":
            assert any(job["node_list"] in line for line in stderr)
        assert tools.execute("job_accounting", {"job_id": job_id})["gpus"] == job["gpus"]


def test_actions_are_isolated_and_test_only_does_not_create_a_job():
    script = "#!/bin/bash\n#SBATCH --partition=boost_qos_dbg\n#SBATCH --time=00:10:00\nhostname\n"
    initial = tools.execute("job_status", {"job_id": "4315390"})
    assert initial["state"] == "PENDING"
    with tools.mock_session():
        before = tools.execute("list_queue", {})
        tools.execute("submit_job", {"script": script, "test_only": True})
        assert tools.execute("list_queue", {}) == before
        submitted = tools.execute("submit_job", {"script": script})["job_id"]
        status = tools.execute("job_status", {"job_id": submitted})
        assert status["state"] == "PENDING" and status["partition"] == "boost_qos_dbg"
        assert status["submitted_script"] == script and status["time_limit"] is None
        assert "max_rss" not in tools.execute("job_accounting", {"job_id": submitted})
        assert tools.execute("submit_job", {"script": script})["job_id"] != submitted
        assert tools.execute("cancel_job", {"job_id": "4315390"})["cancelled"]
        assert tools.execute("job_status", {"job_id": "4315390"})["state"] == "CANCELLED"
        assert not tools.execute("cancel_job", {"job_id": "4315390"})["cancelled"]
        assert "4315390" not in {j["job_id"] for j in tools.execute("list_queue", {})["jobs"]}
        assert "max_rss" not in tools.execute("job_accounting", {"job_id": "4315390"})
    assert tools.execute("job_status", {"job_id": "4315390"}) == initial


def test_submission_cannot_overwrite_a_previously_observed_terminal_job():
    script = "#!/bin/bash\n#SBATCH --partition=boost_qos_dbg\n#SBATCH --time=00:10:00\nhostname\n"
    prospective = tools.execute("submit_job", {"script": script})["job_id"]
    with tools.mock_session():
        before = tools.execute("job_status", {"job_id": prospective})
        assert before["state"] not in ("PENDING", "RUNNING")
        new_id = tools.execute("submit_job", {"script": script})["job_id"]
        assert new_id != prospective
        assert tools.execute("job_status", {"job_id": prospective}) == before


class ActionTeacher:
    def __init__(self, calls): self.calls = iter(calls)
    def complete(self, *args, **kwargs):
        call = next(self.calls, None)
        return Completion(content="Checking." if call else "The simulated observations are shown above.",
                          tool_calls=[call] if call else None)


def test_real_generation_replay_and_verification_execute_actions_once(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, max_tool_rounds=4)
    script = "#!/bin/bash\n#SBATCH --partition=boost_qos_dbg\nhostname\n"
    with tools.mock_session():
        jid = tools.execute("submit_job", {"script": script})["job_id"]
    calls = [_call("submit_job", script=script), _call("job_status", job_id=jid),
             _call("cancel_job", job_id=jid), _call("job_status", job_id=jid)]
    question = "Submit this script and inspect the synthetic job, then cancel it and confirm: " + script
    convo = generate.tool_trace(ActionTeacher(calls), cfg, "Synthetic local mock contract.", question)
    results = [json.loads(m["content"]) for m in convo if m["role"] == "tool"]
    assert results[1]["state"] == "PENDING" and results[3]["state"] == "CANCELLED"
    row = {"id": "actions", "mode": "call", "question": question, "messages": convo, "mock_version": tools.MOCK_VERSION}
    observed = []
    original = toolcheck.execute
    def counted(name, args):
        observed.append(name)
        return original(name, args)
    monkeypatch.setattr(toolcheck, "execute", counted)
    write_jsonl(cfg.run_dir / "generated.jsonl", [row])
    verify.run(cfg)
    assert observed == [c["function"]["name"] for c in calls]
    assert len(read_jsonl(cfg.run_dir / "verified.jsonl")) == 1
    assert read_jsonl(cfg.run_dir / "rejected.jsonl") == []


def test_concurrent_production_traces_have_independent_state(tmp_path):
    cfg = _cfg(tmp_path, max_tool_rounds=2)
    calls = [_call("cancel_job", job_id="4315390"), _call("job_status", job_id="4315390")]
    def one(_):
        convo = generate.tool_trace(ActionTeacher(calls), cfg, "Synthetic contract.", "Cancel 4315390 and check its state.")
        row = {"mode": "call", "messages": convo}
        assert toolcheck.check_trace(row)["passed"]
        return convo
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, range(12)))
    assert all(row == rows[0] for row in rows)
    assert tools.execute("job_status", {"job_id": "4315390"})["state"] == "PENDING"


def test_partition_guess_is_rejected_but_discovery_is_supported(tmp_path):
    cfg = _cfg(tmp_path, max_tool_rounds=2)
    def checked(calls):
        convo = generate.tool_trace(ActionTeacher(calls), cfg, "Synthetic contract.", "Check GPU availability.")
        return toolcheck.check_trace({"mode": "call", "messages": convo})
    bad = checked([_call("gpu_availability", partition="your_partition")])
    assert bad["ungrounded_partitions"] == ["your_partition"] and not bad["passed"]
    assert checked([_call("partition_info"), _call("gpu_availability", partition="boost_usr_prod")])["passed"]


def test_unknown_mock_version_requires_archived_replay(tmp_path):
    row = {"mode": "none", "mock_version": "unknown-old-fixture", "messages": [
        {"role": "user", "content": "Synthetic question"}, {"role": "assistant", "content": "Synthetic answer"}]}
    result = toolcheck.check_trace(row)
    assert not result["passed"] and "archived implementation" in result["trace_errors"][0]


def test_repository_gold_fixture_provenance_matches_reviewed_implementation():
    evidence = json.loads((ROOT / "data/gold/provenance.json").read_text())["examples"]["tool_trace.json"]["local_fixture_evidence"]
    assert hashlib.sha256((ROOT / evidence["path"]).read_bytes()).hexdigest() == evidence["sha256"]
