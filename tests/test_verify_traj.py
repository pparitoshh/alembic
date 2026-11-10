import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_traj", ROOT / "research/train_multistep_tools/verify_traj.py")
vt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vt)


def call(name, args):
    return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def good():
    return {
        "id": "t1", "family": "debug_failed_job", "question": "Why did job 1234567 fail?",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Why did job 1234567 fail? It ran on boost_usr_prod."},
            {"role": "assistant", "content": "", "tool_calls": [call("job_status", {"job_id": "1234567"})]},
            {"role": "tool", "content": json.dumps({"job_id": "1234567", "state": "FAILED", "mem": "1,000 GB"})},
            {"role": "assistant", "content": "", "tool_calls": [call("read_job_log", {"job_id": "1234567"})]},
            {"role": "tool", "content": "slurmstepd: Exceeded memory limit, 4 GPUs"},
            {"role": "assistant", "content": "Job 1234567 on boost_usr_prod failed: it used 1000 GB on 4 GPUs "
                                             "and went over the memory limit."},
        ],
        "world": {"partitions": ["boost_usr_prod", "dcgp_usr_prod"]},
        "task": {"user_request": "Why did job 1234567 fail?", "expected_tools": ["job_status", "read_job_log"],
                 "needs_tool": True, "multi_step": True},
    }


def reasons(row, prefix):
    r = vt.verify(row)
    return [x for x in r["fail_reasons"] if x.startswith(prefix)]


def test_good_row_passes():
    r = vt.verify(good())
    assert r["pass"] is True and r["fail_reasons"] == []


def test_rule1_schema():
    assert reasons(good(), "schema") == []
    row = good()
    row["messages"][2]["tool_calls"] = [call("job_status", {"job_id": 1234567})]
    assert reasons(row, "schema")
    row = good()
    row["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{bad json"
    assert reasons(row, "schema")
    row = good()
    row["messages"][4]["tool_calls"] = [call("read_job_log", {"job_id": "1", "stream": "both"})]
    assert reasons(row, "schema")


def test_rule2_leak():
    assert reasons(good(), "leak") == []
    row = good()
    row["messages"][2]["content"] = "I will use job_status first."
    assert reasons(row, "leak")
    row = good()
    row["messages"][-1]["content"] += ' <tool_call>{"name": "x", "arguments": {}}</tool_call>'
    assert reasons(row, "leak")


def test_rule3_grounding():
    assert reasons(good(), "grounding") == []
    row = good()
    row["messages"][-1]["content"] = "Job 7654321 failed on dcgp_usr_prod with 64 GB."
    got = reasons(row, "grounding")
    assert any("7654321" in g for g in got)
    assert any("dcgp_usr_prod" in g for g in got)
    assert any("64" in g for g in got)


def test_rule4_final():
    assert reasons(good(), "final") == []
    row = good()
    row["messages"][-1]["content"] = "Let me check the log for job 1234567."
    assert reasons(row, "final")
    row = good()
    row["messages"] = row["messages"][:-1]
    assert reasons(row, "final")
    row = good()
    extra = row["messages"][2:4] * 6
    row["messages"] = row["messages"][:2] + extra + row["messages"][2:]
    assert any("tool calls" in g for g in reasons(row, "final"))


def test_rule5_coverage():
    assert reasons(good(), "coverage") == []
    row = good()
    row["messages"] = row["messages"][:4] + row["messages"][-1:]
    row["messages"][-1]["content"] = "Job 1234567 failed."
    got = reasons(row, "coverage")
    assert any("multi_step" in g for g in got)
    row = good()
    row["messages"] = row["messages"][:2] + [{"role": "assistant", "content": "Please give the job ID."}]
    got = reasons(row, "coverage")
    assert any("needs_tool" in g for g in got) and any("recall" in g for g in got)


def test_rule6_action():
    assert reasons(good(), "action") == []
    row = good()
    row["messages"][4]["tool_calls"] = [call("cancel_job", {"job_id": "1234567"})]
    assert reasons(row, "action")
    row["family"] = "cancel_job"
    assert reasons(row, "action") == []


def cancel_row(result):
    row = good()
    row["family"] = "cancel"
    row["task"].update(user_request="Cancel job 1234567", expected_tools=["cancel_job"], multi_step=False)
    row["messages"] = row["messages"][:2] + [
        {"role": "assistant", "content": "", "tool_calls": [call("cancel_job", {"job_id": "1234567"})]},
        {"role": "tool", "content": json.dumps(result)},
        {"role": "assistant", "content": "Job 1234567 is cancelled."}]
    return row


def test_rule7_claim():
    assert reasons(cancel_row({"ok": True}), "claim") == []
    assert reasons(cancel_row({"error": "permission denied"}), "claim")
    row = good()
    row["messages"][-1]["content"] = "Job 1234567 was cancelled."
    assert reasons(row, "claim")


def test_cli(tmp_path, capsys):
    bad = copy.deepcopy(good())
    bad["family"] = "other"
    bad["messages"][-1]["content"] = "Let me check."
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps(good()) + "\n" + json.dumps(bad) + "\n")
    out = tmp_path / "out.jsonl"
    vt.main([str(src), str(out)])
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["pass"] for r in rows] == [True, False]
    text = capsys.readouterr().out
    assert "debug_failed_job" in text and "final" in text and "ALL" in text


def test_coverage_with_three_expected_tools():
    row = good()  # calls job_status and read_job_log: 2 of the 3 expected tools
    row["task"]["expected_tools"] = ["job_status", "read_job_log", "job_accounting"]
    assert reasons(row, "coverage") == []  # recall 2/3 >= 0.5 and 2 distinct tools
    row["messages"] = row["messages"][:4] + row["messages"][-1:]  # only job_status
    row["messages"][-1]["content"] = "Job 1234567 failed."
    got = reasons(row, "coverage")
    assert any("recall" in g for g in got) and any("multi_step" in g for g in got)


def test_cli_first_pass_per_task(tmp_path, capsys):
    answer = good()["messages"][-1]["content"]
    rows = []
    for aid, sample, content in [
            ("t1", 0, "Let me check the log for job 1234567."),  # fails
            ("t1", 1, answer),
            ("t1", 2, answer),
            ("t2", 0, "Let me check the log for job 1234567.")]:  # fails, and no passing sample at all
        row = good()
        row["id"], row["sample"] = aid, sample
        row["messages"][-1]["content"] = content
        rows.append(row)
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    vt.main([str(src), str(tmp_path / "all.jsonl")])
    all_rows = [json.loads(x) for x in (tmp_path / "all.jsonl").read_text().splitlines()]
    assert [(r["id"], r["sample"], r["pass"]) for r in all_rows] == [
        ("t1", 0, False), ("t1", 1, True), ("t1", 2, True), ("t2", 0, False)]  # the flag stays off: every row
    vt.main([str(src), str(tmp_path / "first.jsonl"), "--first-pass-per-task"])
    kept = [json.loads(x) for x in (tmp_path / "first.jsonl").read_text().splitlines()]
    assert [(r["id"], r["sample"]) for r in kept] == [("t1", 1)]
    assert "kept: 1 of 4" in capsys.readouterr().out


def test_cli_first_pass_per_task_picks_the_smallest_sample(tmp_path, capsys):
    answer = good()["messages"][-1]["content"]
    rows = []
    for sample in (2, 1):  # a later sample arrives first in the file (concurrent collection)
        row = good()
        row["id"], row["sample"] = "t1", sample
        row["messages"][-1]["content"] = answer
        rows.append(row)
    src = tmp_path / "in.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    vt.main([str(src), str(tmp_path / "first.jsonl"), "--first-pass-per-task"])
    kept = [json.loads(x) for x in (tmp_path / "first.jsonl").read_text().splitlines()]
    assert [(r["id"], r["sample"]) for r in kept] == [("t1", 1)]
