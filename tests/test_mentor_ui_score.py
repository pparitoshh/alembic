"""Scoring in mentor_ui.py: session matching and the per-case metrics, with fake zstd session logs (no dsh, no browser)."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dsh_mentor_ui", ROOT / "research" / "dsh_harness" / "mentor_ui.py")
mui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mui)

try:
    import zstandard  # noqa: F401
except ImportError:
    pass

# the script always decompresses with the zstd CLI, so the tests need it even with the zstandard module
pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="needs the zstd CLI")


def write_zstd(path, text):
    """A zstd-compressed file, via the zstandard module or the zstd CLI."""
    try:
        import zstandard

        path.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    except ImportError:
        p = subprocess.run(["zstd", "-q", "-c", "-"], input=text.encode(), capture_output=True, check=True)
        path.write_bytes(p.stdout)


def session(sessions_dir, sid, created_ms, events):
    lines = [{"type": "session", "version": 4, "id": f"session-{sid}", "createdAt": created_ms}] + events
    d = sessions_dir / f"session-{sid}"
    d.mkdir(parents=True)
    write_zstd(d / "session.v4.jsonl.zstd", "\n".join(json.dumps(l) for l in lines) + "\n")


def text_msg(text):
    return {"type": "assistant/message", "data": {"message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}}


def call_msg(name):
    return {"type": "assistant/message",
            "data": {"message": {"role": "assistant", "content": [{"type": "tool-call", "name": name, "arguments": "{}"}]}}}


def tool_call(name):
    return {"type": "tool/call", "data": {"name": name, "arguments": "{}"}}


def tool_result(text, error=False):
    msg = {"role": "tool", "content": [{"type": "text", "text": text}]}
    if error:
        msg["isError"] = True
    return {"type": "tool/result", "data": {"message": msg}}


def case(cid, actions=()):
    return {"id": cid, "user_request": "do the thing", "expected_actions": list(actions)}


def run_entry(cid, start, end):
    return {"id": cid, "start": start, "end": end, "visible": "answer"}


def do_score(tmp_path, cases, runs, sessions_dir):
    """Score runs against fake sessions; returns the parsed scores.json."""
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "usecases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    (out / "runs.json").write_text(json.dumps(runs))
    mui.score(str(tmp_path / "usecases.jsonl"), str(out), str(sessions_dir))
    return json.loads((out / "scores.json").read_text())


def test_session_matching_by_time(tmp_path):
    sd = tmp_path / "sessions"
    # session inside the run window [start-5, end]: created 995 s for a run 1000..1040
    session(sd, "inwindow", 995_000, [call_msg("mcp__hpc__partition_info"), tool_call("mcp__hpc__partition_info")])
    # session outside the window: too late
    session(sd, "late", 1050_000, [tool_call("mcp__hpc__list_queue")])
    scores = do_score(tmp_path, [case("c1", ["partition_info"])], [run_entry("c1", 1000, 1040)], sd)
    s = scores[0]
    assert s["session"] == "session-inwindow"
    assert s["tool_calls"] == ["partition_info"]
    assert s["expected_hit"] == ["partition_info"] and s["recall"] == 1.0


def test_duplicate_tool_calls_count_per_call(tmp_path):
    sd = tmp_path / "sessions"
    session(sd, "s1", 995_000, [call_msg("mcp__hpc__job_status"), tool_call("mcp__hpc__job_status"),
                                tool_result("ok"), call_msg("mcp__hpc__job_status"), tool_call("mcp__hpc__job_status")])
    scores = do_score(tmp_path, [case("c1", ["job_status"])], [run_entry("c1", 1000, 1040)], sd)
    s = scores[0]
    assert s["tool_calls"] == ["job_status", "job_status"]
    assert scores[-1]["totals"]["real_tool_calls"] == 2
    assert s["expected_hit"] == ["job_status"] and s["recall"] == 1.0


def test_session_folder_starting_with_dashes(tmp_path):
    # real session folders start with "--"; the path must stay literal for zstd (passed after "--")
    sd = tmp_path / "--home-paritosh-Documents-deepseek-harness-default-workspace--"
    session(sd, "s1", 995_000, [call_msg("mcp__hpc__partition_info"), tool_call("mcp__hpc__partition_info")])
    scores = do_score(tmp_path, [case("c1", ["partition_info"])], [run_entry("c1", 1000, 1040)], sd)
    assert scores[0]["tool_calls"] == ["partition_info"] and scores[0]["recall"] == 1.0


def test_fake_tool_calls_in_text(tmp_path):
    sd = tmp_path / "sessions"
    events = [text_msg('Let me show the call: {"name": "mcp__hpc__partition_info", "arguments": {}}'),
              text_msg("You can run mcp__hpc__list_queue yourself."),
              tool_call("mcp__hpc__job_status")]
    session(sd, "s1", 995_000, events)
    scores = do_score(tmp_path, [case("c1")], [run_entry("c1", 1000, 1040)], sd)
    s = scores[0]
    assert s["fake_tool_calls"] == 2 and s["tool_calls"] == ["job_status"]


def test_recall_with_hidden_submit_job(tmp_path):
    sd = tmp_path / "sessions"
    session(sd, "s1", 995_000, [call_msg("mcp__hpc__partition_info"), tool_call("mcp__hpc__partition_info")])
    cases = [case("c1", ["partition_info", "submit_job"]), case("c2", ["submit_job"])]
    runs = [run_entry("c1", 1000, 1040), run_entry("c2", 2000, 2040)]
    scores = do_score(tmp_path, cases, runs, sd)
    assert scores[0]["expected_visible"] == ["partition_info"] and scores[0]["recall"] == 1.0
    # only hidden actions expected: nothing visible, recall null
    assert scores[1]["expected_visible"] == [] and scores[1]["recall"] is None


def test_stopped_early(tmp_path):
    sd = tmp_path / "sessions"
    session(sd, "announce", 995_000, [text_msg("The setup looks fine so far. I will now check the partition limits.")])
    session(sd, "answer", 1995_000, [text_msg("Here is the full job script with all options you need. " * 6)])
    session(sd, "thencall", 2995_000, [text_msg("I will now check the partition limits."),
                                       call_msg("mcp__hpc__partition_info"), tool_call("mcp__hpc__partition_info")])
    cases = [case("c1"), case("c2"), case("c3")]
    runs = [run_entry("c1", 1000, 1040), run_entry("c2", 2000, 2040), run_entry("c3", 3000, 3040)]
    scores = do_score(tmp_path, cases, runs, sd)
    assert [s["stopped_early"] for s in scores[:3]] == [True, False, False]


def test_missing_session(tmp_path):
    sd = tmp_path / "sessions"
    session(sd, "late", 1050_000, [text_msg("an answer " * 100)])
    scores = do_score(tmp_path, [case("c1", ["partition_info"])], [run_entry("c1", 1000, 1040)], sd)
    s = scores[0]
    assert s["session"] is None and s["tool_calls"] == [] and s["recall"] == 0.0
    assert s["empty_answer"] is True and s["stopped_early"] is False


def test_tool_errors_and_empty_answer(tmp_path):
    sd = tmp_path / "sessions"
    session(sd, "s1", 995_000, [call_msg("mcp__hpc__job_status"), tool_call("mcp__hpc__job_status"),
                                tool_result("boom", error=True), text_msg("short")])
    scores = do_score(tmp_path, [case("c1")], [run_entry("c1", 1000, 1040)], sd)
    s = scores[0]
    assert s["tool_errors"] == 1 and s["empty_answer"] is True and s["answer_chars"] == len("short")


def test_totals_and_output_files(tmp_path):
    sd = tmp_path / "sessions"
    session(sd, "s1", 995_000, [call_msg("mcp__hpc__partition_info"), tool_call("mcp__hpc__partition_info"),
                                text_msg('Try mcp__hpc__list_queue for the queue.')])
    cases = [case("c1", ["partition_info", "submit_job"]), case("c2", ["partition_info"])]
    runs = [run_entry("c1", 1000, 1040), run_entry("c2", 2000, 2040)]
    scores = do_score(tmp_path, cases, runs, sd)
    assert scores[-1]["totals"] == {"cases": 2, "cases_with_real_tools": 1, "real_tool_calls": 1,
                                    "mean_recall": 0.5, "fake_tool_call_cases": 1, "tool_error_cases": 0,
                                    "empty_answer_cases": 2, "stopped_early_cases": 0, "missing_session_cases": 1}
    md = (tmp_path / "out" / "scores.md").read_text()
    assert "| c1 |" in md and "| c2 |" in md and "| cases | 2 |" in md
    assert scores[-2]["must_not"] is None and scores[-2]["key_points"] is None
