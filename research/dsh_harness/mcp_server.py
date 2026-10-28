"""Local MCP server (stdio) for the 17 Slurm tools of `distillkit.slurm_live`, so DeepSeek Harness (dsh) and Hermes
Agent can load them. Standard library only, plus distillkit.

Transport: JSON-RPC 2.0, one JSON message per line on stdin and stdout. Logs go to stderr only.

Safety: the approval modes of dsh and Hermes do not cover MCP tools, so this server is the safety gate. Actions that
change the cluster (submit_job without test_only, cancel_job, update_job, hold_job, release_job, requeue_job) go
through the `confirm(name, args)` callback of SlurmLive. The callback returns True only if HPC_ALLOW_ACTIONS is "1".
Otherwise it declines and nothing changes on the cluster. Without the flag the action tools (except submit_job, which
stays listed for test_only) are also hidden from tools/list, and every refused call answers with a clear "do not
retry" message, so a small model does not burn rounds on a tool it cannot use.

Environment:
  HPC_CLUSTER        cluster name from the clusters file (default empty: use HPC_SSH/HPC_REMOTE_SETUP directly)
  HPC_CLUSTERS_FILE  path of the clusters file (default ~/.config/hpc-assistant/clusters.yaml)
  HPC_SSH            login node for ssh (default: the cluster profile's 'ssh', else run the commands locally)
  HPC_REMOTE_SETUP   shell snippet to run before each remote command (default: the profile's 'remote_setup');
                     both HPC_SSH and HPC_REMOTE_SETUP win over the profile when set
  HPC_ALLOW_ACTIONS  "1" to allow actions (default "0")

Run: PYTHONPATH=src python research/dsh_harness/mcp_server.py
"""
import json
import os
import sys

from distillkit.clusters import CLUSTERS_FILE, get_cluster
from distillkit.slurm_live import SlurmLive, openai_tools
from distillkit.tools import ToolError

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "hpc-slurm-tools", "version": "0.1.0"}
# Job-changing tools hidden from tools/list while HPC_ALLOW_ACTIONS != "1" (submit_job stays: test_only works)
HIDDEN_TOOLS = ("cancel_job", "update_job", "hold_job", "release_job", "requeue_job")
# The reply for a refused action (declined by confirm, or a call to a hidden tool): tell the model to stop retrying
ACTIONS_DISABLED = ("Job actions are disabled on this server. Do not retry. Tell the user what you wanted to do and "
                    "show them the command or script, so they can run it themselves.")
# Added to the submit_job description while actions are disabled (test_only=true still works)
TEST_ONLY_NOTE = " Only test_only=true works on this server: actions are disabled, so a real submission is refused."
_DECLINED = "the user declined this action"  # the decline result of SlurmLive.execute (kept there for other callers)


def log(*parts):
    print("[hpc-slurm-tools]", *parts, file=sys.stderr, flush=True)


def make_slurm(env=None):
    """Build the SlurmLive object from the environment. HPC_CLUSTER picks a profile from the clusters file;
    explicit HPC_SSH / HPC_REMOTE_SETUP values win over the profile. Raises ToolError for a bad file or name."""
    env = os.environ if env is None else env
    allow = env.get("HPC_ALLOW_ACTIONS", "0") == "1"

    def confirm(name, args):
        log(f"action {name} {'allowed' if allow else 'declined'} (HPC_ALLOW_ACTIONS)")
        return allow

    profile = get_cluster(env["HPC_CLUSTER"], env.get("HPC_CLUSTERS_FILE") or CLUSTERS_FILE) \
        if env.get("HPC_CLUSTER") else {}
    return SlurmLive(ssh=env.get("HPC_SSH") or profile.get("ssh"),
                     remote_setup=env.get("HPC_REMOTE_SETUP") or profile.get("remote_setup") or None,
                     job_defaults=profile.get("defaults"), confirm=confirm)


def list_tools(allow=False):
    """The MCP tool list. Without the allow flag the job-changing tools stay hidden and submit_job says that only
    test_only=true works, so the model does not plan calls it cannot make."""
    tools = [{"name": t["function"]["name"], "description": t["function"]["description"],
              "inputSchema": t["function"]["parameters"]} for t in openai_tools()]
    if allow:
        return tools
    return [{**t, "description": t["description"] + TEST_ONLY_NOTE} if t["name"] == "submit_job" else t
            for t in tools if t["name"] not in HIDDEN_TOOLS]


def call_tool(slurm, params):
    """Run one tools/call. Every failure comes back as isError true, never as an exception."""
    try:
        # A null argument from the model means "not given": drop it, so an optional argument passes
        # and a required one fails with the normal missing-argument error.
        arguments = {k: v for k, v in (params.get("arguments") or {}).items() if v is not None}
        result = slurm.execute(params.get("name"), arguments)
        if isinstance(result, dict) and result.get("error") == _DECLINED:
            return {"content": [{"type": "text", "text": ACTIONS_DISABLED}], "isError": True}
        return {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False}
    except ToolError as e:
        text = str(e)
    except Exception as e:  # keep the loop alive
        text = f"{type(e).__name__}: {e}"
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle_message(msg, slurm, allow=False):
    """One decoded JSON-RPC message in, one reply dict out (None for notifications). The allow flag is the
    HPC_ALLOW_ACTIONS value this server started with: it decides which tools the list shows."""
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
        result = {"tools": list_tools(allow)}
    elif method == "tools/call":
        result = call_tool(slurm, params if isinstance(params, dict) else {})
    else:
        return _error(msg_id, -32601, f"method not found: {method}")
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def handle_line(line, slurm, allow=False):
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        return _error(None, -32700, f"parse error: {e}")
    return handle_message(msg, slurm, allow)


def main():
    try:
        slurm = make_slurm()
    except ToolError as e:  # a bad clusters file or an unknown cluster: never start with wrong settings
        log(str(e))
        sys.exit(1)
    allow = os.environ.get("HPC_ALLOW_ACTIONS", "0") == "1"
    cluster = os.environ.get("HPC_CLUSTER") or "(none)"
    log(f"ready; cluster = {cluster}; ssh = {slurm.ssh or '(local)'}; "
        f"remote setup = {'yes' if slurm.remote_setup else 'no'}; actions {'allowed' if allow else 'hidden'}")
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle_line(line, slurm, allow)
        if reply is not None:
            print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
