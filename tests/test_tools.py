import json
from pathlib import Path

import pytest
import yaml

from distillkit import verify
from distillkit.config import Config
from distillkit.generate import question_jobs, to_api, tool_trace
from distillkit.io import read_jsonl, write_jsonl
from distillkit.teacher import Completion
from distillkit.toolcheck import ast_match, check_trace, parse_hermes, score_tool_item
from distillkit.tools import SCHEMAS, TOOLS, ToolError, execute, validate_call

ROOT = Path(__file__).parent.parent


def _cfg(tmp_path, **generate) -> Config:
    d = yaml.safe_load((ROOT / "configs/toy_qdora.yaml").read_text())
    d["run_dir"] = str(tmp_path / "run")
    d["seeds"]["dir"] = str(ROOT / "data/seeds")
    d["verify"]["flag_list"] = str(ROOT / "data/slurm_flags.txt")
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
