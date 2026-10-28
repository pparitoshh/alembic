"""Local MCP server (stdio) for the 16 Slurm tools of `distillkit.slurm_live`, so DeepSeek Harness (dsh) and Hermes
Agent can load them. Standard library only, plus distillkit.

Transport: JSON-RPC 2.0, one JSON message per line on stdin and stdout. Logs go to stderr only.

Safety: the approval modes of dsh and Hermes do not cover MCP tools, so this server is the safety gate. Actions that
change the cluster (submit_job without test_only, cancel_job, update_job, hold_job, release_job, requeue_job) go
through the `confirm(name, args)` callback of SlurmLive. The callback returns True only if HPC_ALLOW_ACTIONS is "1".
Otherwise it declines and nothing changes on the cluster.

Environment:
  HPC_SSH            login node for ssh (default empty: run the commands locally)
  HPC_REMOTE_SETUP   shell snippet to run before each remote command, e.g. "module load slurm" (default empty: none)
  HPC_ALLOW_ACTIONS  "1" to allow actions (default "0")

Run: PYTHONPATH=src python research/dsh_harness/mcp_server.py
"""
import json
import os
import sys

from distillkit.slurm_live import SlurmLive, openai_tools
from distillkit.tools import ToolError

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "hpc-slurm-tools", "version": "0.1.0"}


def log(*parts):
    print("[hpc-slurm-tools]", *parts, file=sys.stderr, flush=True)


def make_slurm(env=None):
    """Build the SlurmLive object from the environment."""
    env = os.environ if env is None else env
    allow = env.get("HPC_ALLOW_ACTIONS", "0") == "1"

    def confirm(name, args):
        log(f"action {name} {'allowed' if allow else 'declined'} (HPC_ALLOW_ACTIONS)")
        return allow

    return SlurmLive(ssh=env.get("HPC_SSH") or None, remote_setup=env.get("HPC_REMOTE_SETUP") or None, confirm=confirm)


def list_tools():
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "inputSchema": t["function"]["parameters"]} for t in openai_tools()]


def call_tool(slurm, params):
    """Run one tools/call. Every failure comes back as isError true, never as an exception."""
    try:
        # A null argument from the model means "not given": drop it, so an optional argument passes
        # and a required one fails with the normal missing-argument error.
        arguments = {k: v for k, v in (params.get("arguments") or {}).items() if v is not None}
        result = slurm.execute(params.get("name"), arguments)
        return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
    except ToolError as e:
        text = str(e)
    except Exception as e:  # keep the loop alive
        text = f"{type(e).__name__}: {e}"
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(msg, slurm):
    """One decoded JSON-RPC message in, one reply dict out (None for notifications)."""
    if not isinstance(msg, dict):
        return _error(None, -32600, "invalid request")
    method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if "id" not in msg:  # notification: never gets a reply
        return None
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": list_tools()}
    elif method == "tools/call":
        result = call_tool(slurm, params if isinstance(params, dict) else {})
    else:
        return _error(msg_id, -32601, f"method not found: {method}")
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def handle_line(line, slurm):
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        return _error(None, -32700, f"parse error: {e}")
    return handle_message(msg, slurm)


def main():
    slurm = make_slurm()
    log(f"ready; ssh = {slurm.ssh or '(local)'}; remote setup = {'yes' if slurm.remote_setup else 'no'}")
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle_line(line, slurm)
        if reply is not None:
            print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
