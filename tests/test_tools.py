import json
from pathlib import Path

import pytest
import yaml

from distillkit import verify
from distillkit.config import Config
from distillkit.generate import question_jobs, to_api, tool_trace
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import final_answer
from distillkit.teacher import Completion
from distillkit.toolcheck import ast_match, check_trace, parse_hermes, score_tool_item
from distillkit.tools import PARTITIONS, SCHEMAS, TOOLS, ToolError, execute, validate_call

ROOT = Path(__file__).parent.parent


def _cfg(tmp_path, **generate) -> Config:
    d = yaml.safe_load((ROOT / "configs/toy_qdora.yaml").read_text())
    d["run_dir"] = str(tmp_path / "run")
    d["seeds"]["dir"] = str(ROOT / "data/seeds")
    d["eval"]["file"] = str(ROOT / "data/eval/eval_all.jsonl")
    d["generate"] |= generate
    return Config.model_validate(d)


def _call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


# --- tools -------------------------------------------------------------------------------------


def test_schemas_are_well_formed():
    assert 5 <= len(SCHEMAS) <= 10
    for s in SCHEMAS:
        p = s["parameters"]
        assert s["description"] and p["type"] == "object"
        assert set(p["required"]) <= set(p["properties"])


def test_mock_cluster_is_deterministic_and_consistent():
    assert execute("job_status", {"job_id": "4718207"}) == execute("job_status", '{"job_id": "4718207"}')
    for jid in map(str, range(4718200, 4718260)):
        job = execute("job_status", {"job_id": jid})
        if job["state"] not in ("PENDING", "TIMEOUT"):
            assert job["elapsed"] < job["time_limit"].rjust(8, "0") or "-" in job["time_limit"]
        acct = execute("job_accounting", {"job_id": jid})
        if job["state"] == "OUT_OF_MEMORY" and job["gpus"]:
            assert acct["gpu_mem_peak"] == "63.4GiB" and acct["max_rss"] != "494G"


@pytest.mark.parametrize(
    "name, args, msg",
    [
        ("squeue", {}, "unknown tool"),
        ("job_status", {}, "missing"),
        ("job_status", {"job_id": "1", "verbose": True}, "unknown argument"),
        ("job_status", {"job_id": 42}, "should be string"),
        ("read_job_log", {"job_id": "1", "stream": "both"}, "must be one of"),
        ("read_job_log", {"job_id": "1", "tail_lines": True}, "should be integer"),
    ],
)
def test_validate_call_rejects(name, args, msg):
    with pytest.raises(ToolError, match=msg):
        validate_call(name, args)


def test_execute_errors():
    with pytest.raises(ToolError, match="invalid job id"):
        execute("job_status", {"job_id": "my-job"})
    with pytest.raises(ToolError, match="not valid JSON"):
        execute("job_status", "{job_id: 1")
    assert "error" in execute("gpu_availability", {"partition": "nope"})  # cluster-side error, valid call


def test_submit_job_lints_the_script():
    ok = execute("submit_job", {"script": "#!/bin/bash\n#SBATCH --time=01:00:00\nsrun hostname"})
    assert ok["submitted"] and ok["job_id"].isdigit()
    bad = execute("submit_job", {"script": "#!/bin/bash\n#SBATCH --gpu-count=2\nsrun hostname"})
    assert "unrecognized option '--gpu-count'" in bad["error"]
    part = execute("submit_job", {"script": "#!/bin/bash\n#SBATCH --partition=gpu\nsrun hostname"})
    assert part["error"] == "sbatch: error: invalid partition specified: gpu"


def test_gpu_snapshot_capacity_and_idle_nodes_are_consistent():
    snapshot = execute("gpu_availability", {})
    assert snapshot == execute("gpu_availability", {})
    for row in snapshot["partitions"]:
        part = PARTITIONS[row["partition"]]
        assert 0 <= row["idle_nodes"] <= part["nodes"]
        assert row["idle_nodes"] * part["gpus_per_node"] <= row["gpus_free"] <= row["gpus_total"]
        assert execute("gpu_availability", {"partition": row["partition"]})["partitions"] == [row]


@pytest.mark.parametrize("directive,partition", [
    ("--partition=boost_qos_dbg", "boost_qos_dbg"), ("--partition dcgp_usr_prod", "dcgp_usr_prod"),
    ("-p boost_qos_lprod", "boost_qos_lprod"), ("", "boost_usr_prod"),
])
def test_test_only_result_respects_requested_partition_without_inventing_cpus(directive, partition):
    script = "#!/bin/bash\n" + (f"#SBATCH {directive}\n" if directive else "") + "#SBATCH --cpus-per-task=2\nhostname\n"
    result = execute("submit_job", {"script": script, "test_only": True})
    assert result["valid"] and "submitted" not in result
    assert result["message"].endswith(f"in partition {partition}")
    assert "32 processors" not in result["message"]
    assert result["validation_scope"] == ["flag_names", "bash_syntax", "partition_name"]


# --- toolcheck ---------------------------------------------------------------------------------


def test_parse_hermes():
    text, calls = parse_hermes('Checking.\n<tool_call>\n{"name": "job_status", "arguments": {"job_id": "7"}}\n</tool_call>')
    assert text == "Checking." and calls == [{"name": "job_status", "arguments": {"job_id": "7"}}]
    _, calls = parse_hermes("<tool_call>\n{broken\n</tool_call>")
    assert calls[0]["name"] is None
    assert parse_hermes("No tools needed.") == ("No tools needed.", [])


def test_ast_match():
    exp = {"name": "read_job_log", "arguments": {"job_id": ["4718206"], "stream": ["stderr", ""], "tail_lines": [50]}}
    assert ast_match({"name": "read_job_log", "arguments": {"job_id": "4718206", "tail_lines": 50}}, exp)
    assert ast_match({"name": "read_job_log", "arguments": '{"job_id": "4718206", "stream": "STDERR", "tail_lines": 50}'}, exp)
    assert not ast_match({"name": "read_job_log", "arguments": {"job_id": "4718206"}}, exp)  # tail_lines required
    assert not ast_match({"name": "read_job_log", "arguments": {"job_id": "4718206", "tail_lines": 50, "x": 1}}, exp)
    assert not ast_match({"name": "job_status", "arguments": {"job_id": "4718206"}}, exp)
    alts = [{"name": "job_status", "arguments": {"job_id": ["1"]}}, {"name": "job_accounting", "arguments": {"job_id": ["1"]}}]
    assert ast_match({"name": "job_accounting", "arguments": {"job_id": "1"}}, alts)
    assert ast_match({"name": "submit_job", "arguments": {"script": "anything"}}, {"name": "submit_job", "arguments": {"script": ["*"]}})


def test_score_tool_item():
    item = {"question": "Is job 4718207 running?", "expect": "call", "expected_call": {"name": "job_status", "arguments": {"job_id": ["4718207"]}}}
    good = score_tool_item('<tool_call>\n{"name": "job_status", "arguments": {"job_id": "4718207"}}\n</tool_call>', item)
    assert good == {"decision": "call", "decision_ok": True, "grounded": True, "valid": True, "exec_ok": True, "ast_ok": True}
    bad = score_tool_item('<tool_call>\n{"name": "job_status", "arguments": {"id": "4718207"}}\n</tool_call>', item)
    assert not bad["valid"] and not bad["exec_ok"] and not bad["ast_ok"]
    assert score_tool_item("Which job id?", {"question": "Why did my job fail?", "expect": "no_call"}) == {"decision": "no_call", "decision_ok": True}
    ask = {"question": "Why did my job fail?", "expect": "no_call", "allow_lookup": True}
    lookup = '<tool_call>\n{"name": "list_queue", "arguments": {}}\n</tool_call>'
    assert score_tool_item(lookup, ask)["decision_ok"]
    assert not score_tool_item(lookup, {**ask, "allow_lookup": False})["decision_ok"]
    guess = score_tool_item('<tool_call>\n{"name": "job_status", "arguments": {"job_id": "4718209"}}\n</tool_call>', ask)
    assert not guess["decision_ok"] and not guess["grounded"]


def _trace(mode, *calls, question="Is job 4718207 running?", final="Done."):
    """Transcript with one call per turn; each tool result is what the mock really returns."""
    msgs = [{"role": "user", "content": question}]
    for c in calls:
        try:
            result = execute(c["function"]["name"], c["function"]["arguments"])
        except ToolError as e:
            result = {"error": str(e)}
        msgs += [{"role": "assistant", "content": "", "tool_calls": [c]}, {"role": "tool", "content": json.dumps(result)}]
    return {"mode": mode, "messages": [*msgs, {"role": "assistant", "content": final}]}


def test_check_trace():
    assert check_trace(_trace("call", _call("job_status", job_id="4718207")))["passed"]
    assert check_trace(_trace("ask", question="Why did my job fail?", final="Which job id?"))["passed"]


@pytest.mark.parametrize("ending", ["tool", "call"])
def test_unfinished_trace_is_not_a_final_answer(ending):
    row = _trace("call", _call("job_status", job_id="4718207"))
    row["messages"][1]["content"] = "Let me check."
    row["messages"] = row["messages"][:-1 if ending == "tool" else -2]
    assert final_answer(row) == ""
    assert not check_trace(row)["passed"]


@pytest.mark.parametrize("fault", ["missing", "orphan", "wrong_result", "invalid_json", "extra_user", "extra_answer", "malformed_call"])
def test_trace_structure_and_result_replay(fault):
    row = _trace("call", _call("job_status", job_id="4718207"))
    msgs = row["messages"]
    if fault == "missing":
        del msgs[2]
    elif fault == "orphan":
        msgs.insert(1, {"role": "tool", "content": "{}"})
    elif fault == "wrong_result":
        msgs[2]["content"] = json.dumps({"state": "FABRICATED", "job_id": "4718207"})
    elif fault == "invalid_json":
        msgs[2]["content"] = "not JSON"
    elif fault == "extra_user":
        msgs.insert(3, {"role": "user", "content": "Invented follow-up"})
    elif fault == "extra_answer":
        msgs.append({"role": "assistant", "content": "Another final answer."})
    else:
        msgs[1]["tool_calls"] = [{"function": []}]
    assert not check_trace(row)["passed"]


def test_parallel_results_must_match_call_order():
    calls = [_call("job_status", job_id="4718207"), _call("job_accounting", job_id="4718207")]
    results = [json.dumps(execute(c["function"]["name"], c["function"]["arguments"]), indent=2, sort_keys=True) for c in calls]
    row = {"mode": "call", "messages": [
        {"role": "user", "content": "Check job 4718207."},
        {"role": "assistant", "content": "Checking.", "tool_calls": calls},
        *[{"role": "tool", "content": result} for result in results],
        {"role": "assistant", "content": "Here is the result."},
    ]}
    assert check_trace(row)["passed"]
    row["messages"][2:4] = reversed(row["messages"][2:4])
    assert check_trace(row)["result_errors"]


def test_fabricated_result_cannot_ground_a_later_job_id():
    row = _trace("ask", _call("list_queue"), _call("job_status", job_id="9999999"), question="Check my job.")
    row["messages"][2]["content"] = '{"jobs": [{"job_id": "9999999"}]}'
    checked = check_trace(row)
    assert checked["result_errors"] and checked["ungrounded_ids"] == ["9999999"]
    assert not checked["passed"]


def test_shadowed_result_fields_cannot_smuggle_a_job_id():
    row = _trace("ask", _call("list_queue"), _call("job_status", job_id="9999999"), question="Check my job.")
    actual = row["messages"][2]["content"]
    row["messages"][2]["content"] = '{"jobs":[{"job_id":"9999999"}], ' + actual[1:]
    checked = check_trace(row)
    assert checked["result_errors"] and checked["ungrounded_ids"] == ["9999999"]
    assert not checked["passed"]


@pytest.mark.parametrize("invalid_calls", [{}, False, 0, ""])
def test_falsey_non_list_calls_are_rejected(invalid_calls):
    row = _trace("ask", final="Which job ID?")
    row["messages"][-1]["tool_calls"] = invalid_calls
    assert not check_trace(row)["passed"]


def test_job_ids_must_be_grounded():
    my_job = "Why is my job stuck?"
    queued = execute("list_queue", {})["jobs"][0]["job_id"]
    # look the job up, then use an id from the tool result: fine, in any mode
    lookup = _trace("ask", _call("list_queue"), _call("job_status", job_id=queued), question=my_job)
    assert check_trace(lookup)["passed"]
    # an id the user never gave and no tool returned: rejected
    invented = check_trace(_trace("ask", _call("job_status", job_id="4718209"), question=my_job))
    assert invented["ungrounded_ids"] == ["4718209"] and not invented["passed"]
    assert check_trace(_trace("call", _call("job_status", job_id="4718208")))["ungrounded_ids"] == ["4718208"]
    # array tasks: grounded through their job id; digits must match whole numbers
    assert check_trace(_trace("call", _call("job_status", job_id="4718207_3")))["passed"]
    assert check_trace(_trace("call", _call("job_status", job_id="471820"), question="job 4718207"))["ungrounded_ids"] == ["471820"]
    assert not check_trace(_trace("call"))["decision_ok"]  # should have called
    assert not check_trace(_trace("none", _call("job_status", job_id="4718207")))["decision_ok"]  # needless call
    assert check_trace(_trace("call", _call("job_status", jobid="1")))["call_errors"]
    assert check_trace(_trace("call", _call("submit_job", script="#!/bin/bash\n#SBATCH --gpu-count=1\n")))["call_errors"]


# --- generation ----------------------------------------------------------------------------------


def test_tool_modes_keep_prose_ids_stable(tmp_path):
    prose_only = question_jobs(_cfg(tmp_path))
    mixed = question_jobs(_cfg(tmp_path, tool_fraction=0.3))
    assert all(j["mode"] == "prose" for j in prose_only)
    assert [j["id"].split("/")[:3] for j in prose_only] == [j["id"].split("/")[:3] for j in mixed]
    share = sum(j["mode"] != "prose" for j in mixed) / len(mixed)
    assert 0.15 < share < 0.45, share
    assert {"call", "ask", "none"} <= {j["mode"] for j in mixed}


def test_to_api_pairs_parallel_calls():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_call("job_status", job_id="1"), _call("job_status", job_id="2")]},
        {"role": "tool", "content": "{}"},
        {"role": "tool", "content": "{}"},
    ]
    api = to_api(msgs)
    assert [c["id"] for c in api[1]["tool_calls"]] == ["call_1", "call_2"]
    assert [m["tool_call_id"] for m in api[2:]] == ["call_1", "call_2"]
    assert json.loads(api[1]["tool_calls"][0]["function"]["arguments"]) == {"job_id": "1"}


class FakeTeacher:
    """Calls job_status once, then answers from the tool result."""

    model = "fake"

    def __init__(self):
        self.requests = []

    def complete(self, messages, tools=None, **_):
        self.requests.append(messages)
        if messages[-1]["role"] == "tool":
            return Completion(content=f"State: {json.loads(messages[-1]['content'])['state']}.")
        return Completion(content="Let me check.", tool_calls=[{"id": "x", "type": "function", "function": {"name": "job_status", "arguments": '{"job_id": "4718207"}'}}])


def test_tool_trace_runs_against_mock(tmp_path):
    t = FakeTeacher()
    convo = tool_trace(t, _cfg(tmp_path), "chunk", "Is 4718207 running?")
    assert [m["role"] for m in convo] == ["user", "assistant", "tool", "assistant"]
    assert convo[1]["tool_calls"][0]["function"]["arguments"] == {"job_id": "4718207"}  # stored as dict, no id
    assert convo[-1]["content"] == f"State: {execute('job_status', {'job_id': '4718207'})['state']}."
    assert "chunk" in t.requests[0][0]["content"]  # grounding goes to the system prompt only


def test_verify_tool_rows(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.run_dir.mkdir()
    meta = {"doc_id": "d", "chunk_id": "d#0", "persona": "p", "task": "debug", "tools": SCHEMAS}
    rows = [
        {"id": "a", "question": "Is job 4718207 running?", **meta, **_trace("call", _call("job_status", job_id="4718207"))},
        {"id": "b", "question": "Why did my job fail, any idea?", **meta, **_trace("ask", question="Why did my job fail, any idea?", final="Which job id?")},
        {"id": "c", "question": "What does the --array option do?", **meta, **_trace("none", _call("list_queue"), question="What does the --array option do?")},
        {"id": "e", "question": "Why is my job pending for so long?", **meta, **_trace("ask", _call("job_status", job_id="4718209"), question="Why is my job pending for so long?")},
        {"id": "d", "question": "Show the log of job 4718206 please", **meta, **_trace("call", _call("read_job_log", job="4718206"), question="Show the log of job 4718206 please")},
    ]
    write_jsonl(cfg.run_dir / "generated.jsonl", rows)
    verify.run(cfg)
    assert [r["id"] for r in read_jsonl(cfg.run_dir / "verified.jsonl")] == ["a", "b"]
    rejected = {r["id"]: r["reject_reason"] for r in read_jsonl(cfg.run_dir / "rejected.jsonl")}
    assert rejected == {"c": "tool_decision", "d": "tool_call_invalid", "e": "tool_ungrounded_id"}


def test_generator_exhaustion_and_corrupt_traces_are_preserved_as_rejections(tmp_path):
    class NeverFinishes(FakeTeacher):
        def complete(self, messages, **kwargs):
            return super().complete([messages[0], {"role": "user", "content": "still checking"}], **kwargs)

    cfg = _cfg(tmp_path, max_tool_rounds=0)
    cfg.run_dir.mkdir()
    exhausted = tool_trace(NeverFinishes(), cfg, "chunk", "Is 4718207 running?")
    missing = _trace("call", _call("job_status", job_id="4718207"), question="Inspect the current state of 4718207.")
    del missing["messages"][2]
    fake = _trace("call", _call("job_accounting", job_id="4718207"), question="Get resource usage for 4718207.")
    fake["messages"][2]["content"] = '{}'
    rows = [
        {"id": "exhausted", "question": "Is 4718207 running?", "mode": "call", "messages": exhausted},
        {"id": "missing", "question": missing["messages"][0]["content"], **missing},
        {"id": "fake", "question": fake["messages"][0]["content"], **fake},
    ]
    write_jsonl(cfg.run_dir / "generated.jsonl", rows)
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir / "verified.jsonl") == []
    rejected = {r["id"]: r for r in read_jsonl(cfg.run_dir / "rejected.jsonl")}
    assert {k: r["reject_reason"] for k, r in rejected.items()} == {
        "exhausted": "empty_or_too_long", "missing": "tool_trace_incomplete_or_out_of_order", "fake": "tool_result_mismatch",
    }
    assert all(rejected[r["id"]]["messages"] == r["messages"] for r in rows)
    assert read_jsonl(cfg.run_dir / "generated.jsonl") == rows


# --- data files ----------------------------------------------------------------------------------


def test_gold_trace_matches_the_mock_cluster():
    msgs = json.loads((ROOT / "data/gold/tool_trace.json").read_text())["messages"]
    assert check_trace({"mode": "call", "messages": msgs})["passed"]
    call = msgs[1]["tool_calls"][0]["function"]
    assert json.loads(msgs[2]["content"]) == execute(call["name"], call["arguments"])  # result is what the mock returns


def test_tool_eval_items():
    rows = read_jsonl(ROOT / "data/eval/eval_tools.jsonl")
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["expect"] in ("call", "no_call") and ("expected_call" in r) == (r["expect"] == "call")
        for e in r.get("expected_call") and (r["expected_call"] if isinstance(r["expected_call"], list) else [r["expected_call"]]) or []:
            assert e["name"] in TOOLS and set(e["arguments"]) <= set(TOOLS[e["name"]]["parameters"]["properties"])
