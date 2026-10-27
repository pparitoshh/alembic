"""MCP server for the Slurm tools: protocol replies and the action gate, with a fake runner (no cluster, no ssh)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from distillkit.slurm_live import SlurmLive

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dsh_mcp_server", ROOT / "research" / "dsh_harness" / "mcp_server.py")
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)

RUNNING = "42|train|RUNNING|None|boost_usr_prod|2|1:02:03|4:00:00|lrdn[0001-0002]|N/A\n"


def fake_slurm(env=None):
    """A server-built SlurmLive whose runner is a recorder; returns (slurm, calls)."""
    calls = []

    def runner(argv, stdin=None):
        calls.append(argv)
        if argv[0] == "id":
            return 0, "me\n", ""
        if argv[0] == "squeue":
            return 0, RUNNING, ""
        return 0, "", ""

    slurm = srv.make_slurm(env or {})
    slurm._runner = runner
    return slurm, calls


def req(method, id=1, **params):
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params}


def test_initialize():
    r = srv.handle_message(req("initialize"), SlurmLive())
    assert r["id"] == 1 and r["result"]["protocolVersion"] == "2025-06-18"
    assert r["result"]["capabilities"] == {"tools": {}}
    assert r["result"]["serverInfo"] == {"name": "hpc-slurm-tools", "version": "0.1.0"}


def test_notification_gets_no_reply():
    assert srv.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}, SlurmLive()) is None


def test_ping():
    assert srv.handle_message(req("ping", id=7), SlurmLive())["result"] == {}


def test_tools_list_has_16_tools():
    tools = srv.handle_message(req("tools/list"), SlurmLive())["result"]["tools"]
    assert len(tools) == 16
    for t in tools:
        assert t["name"] and t["description"] and t["inputSchema"]["type"] == "object"


def test_call_job_status():
    slurm, _ = fake_slurm()
    r = srv.handle_message(req("tools/call", name="job_status", arguments={"job_id": "42"}), slurm)["result"]
    assert r["isError"] is False and r["content"][0]["type"] == "text"
    assert json.loads(r["content"][0]["text"])["state"] == "RUNNING"


def test_call_unknown_tool_and_bad_arguments_are_errors():
    slurm, _ = fake_slurm()
    r = srv.handle_message(req("tools/call", name="nope", arguments={}), slurm)["result"]
    assert r["isError"] is True and "unknown tool" in r["content"][0]["text"]
    r = srv.handle_message(req("tools/call", name="job_status", arguments={}), slurm)["result"]
    assert r["isError"] is True


def test_invalid_json_line():
    assert srv.handle_line("{not json", SlurmLive())["error"]["code"] == -32700


def test_unknown_method():
    r = srv.handle_message(req("bogus/method", id=5), SlurmLive())
    assert r["id"] == 5 and r["error"]["code"] == -32601


def test_action_declined_by_default():
    slurm, calls = fake_slurm({})
    r = srv.handle_message(req("tools/call", name="cancel_job", arguments={"job_id": "42"}), slurm)["result"]
    assert "declined" in r["content"][0]["text"]
    assert not any(c[0] == "scancel" for c in calls)


def test_action_declined_when_flag_is_zero():
    slurm, calls = fake_slurm({"HPC_ALLOW_ACTIONS": "0"})
    srv.handle_message(req("tools/call", name="cancel_job", arguments={"job_id": "42"}), slurm)
    assert not any(c[0] == "scancel" for c in calls)


def test_action_reaches_runner_when_allowed():
    slurm, calls = fake_slurm({"HPC_ALLOW_ACTIONS": "1"})
    r = srv.handle_message(req("tools/call", name="cancel_job", arguments={"job_id": "42"}), slurm)["result"]
    assert r["isError"] is False
    assert ["scancel", "42"] in calls


def test_end_to_end_over_stdio():
    env = {k: v for k, v in os.environ.items() if k not in ("HPC_SSH", "HPC_ALLOW_ACTIONS")}
    env["PYTHONPATH"] = str(ROOT / "src")
    lines = [json.dumps(req("initialize", id=1)), json.dumps(req("tools/list", id=2))]
    p = subprocess.run([sys.executable, str(ROOT / "research" / "dsh_harness" / "mcp_server.py")],
                       input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=20, env=env)
    replies = [json.loads(x) for x in p.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2]
    assert len(replies[1]["result"]["tools"]) == 16
