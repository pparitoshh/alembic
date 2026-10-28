"""Live Slurm backend: command construction and output parsing against recorded Slurm output (no cluster needed)."""
import json
import shlex
from io import StringIO

import pytest

from distillkit import assistant, slurm_live
from distillkit.slurm_live import SlurmLive
from distillkit.tools import SCHEMAS, ToolError


class FakeSlurm:
    """Maps a command (argv[0] plus a distinguishing word) to recorded output; records every call."""

    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def __call__(self, argv, stdin=None):
        self.calls.append((argv, stdin))
        joined = " ".join(argv)
        for key, resp in self.responses.items():
            if all(k in joined for k in key.split("&")):
                return resp
        return 1, "", f"unexpected command: {joined}"


def live(responses, confirm=lambda n, a: True):
    fake = FakeSlurm(responses)
    return SlurmLive(runner=fake, confirm=confirm, workdir="/w/jobs"), fake


def test_live_schemas_are_the_trained_ones_plus_live_tools():
    assert slurm_live.LIVE_SCHEMAS[:len(SCHEMAS)] == SCHEMAS
    assert [t["function"]["name"] for t in slurm_live.openai_tools()][len(SCHEMAS):] == \
        ["job_script", "job_usage", "job_priority", "update_job", "hold_job", "release_job", "requeue_job",
         "run_readonly_command"]


def test_job_status_running_pending_and_finished():
    s, _ = live({"squeue&-j&42": (0, "42|train|RUNNING|None|boost_usr_prod|2|1:02:03|4:00:00|lrdn[0001-0002]|N/A\n", ""),
                 "squeue&-j&43": (0, "43|eval|PENDING|Priority|boost_usr_prod|1|0:00|1:00:00||2026-10-26T10:00:00\n", "")})
    assert s.execute("job_status", {"job_id": "42"}) == {
        "job_id": "42", "name": "train", "state": "RUNNING", "partition": "boost_usr_prod", "nodes": 2,
        "elapsed": "1:02:03", "time_limit": "4:00:00", "node_list": "lrdn[0001-0002]"}
    p = s.execute("job_status", {"job_id": "43"})
    assert p["reason"] == "Priority" and p["est_start"] == "2026-10-26T10:00:00" and "node_list" not in p
    s, _ = live({"squeue": (0, "", ""), "sacct": (0, "44|run|TIMEOUT|0:15|boost_usr_prod|04:00:12|04:00:00|lrdn0007||"
                                                     "14:00:00|8|billing=8,cpu=8,gres/gpu=1,mem=60G,node=1\n", "")})
    f = s.execute("job_status", {"job_id": "44"})
    assert f["state"] == "TIMEOUT" and f["exit_code"] == "0:15" and f["time_limit"] == "04:00:00"


def test_job_accounting_efficiency_memory_and_gpus():
    s, _ = live({"sacct": (0, "7|job|COMPLETED|0:0|p|01:00:00|02:00:00|n1||02:00:00|4|cpu=4,gres/gpu:a100=2,mem=12G,node=1\n"
                              "7.batch|batch|COMPLETED|0:0||01:00:00||n1|3145728K|02:00:00|4|\n", "")})
    a = s.execute("job_accounting", {"job_id": "7"})
    assert a["cpu_efficiency"] == "50%" and a["max_rss"] == "3.0G" and a["gpus"] == 2
    assert a["memory_requested"] == "12.0G" and a["memory_efficiency"] == "25%" and a["time_limit_used"] == "50%"


def test_submit_writes_the_script_then_sbatch_and_test_only():
    s, fake = live({"mkdir": (0, "", ""), "sbatch&--test-only": (0, "", "sbatch: Job 9 to start at 2026-10-26T10:00:00"),
                    "sbatch": (0, "123;cluster\n", "")})
    r = s.execute("submit_job", {"script": "#!/bin/bash\n#SBATCH -t 5\necho hi\n"})
    assert r["submitted"] and r["job_id"] == "123"
    assert fake.calls[0][1] == "#!/bin/bash\n#SBATCH -t 5\necho hi\n"  # the script goes over stdin, never into argv
    t = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n", "test_only": True})
    assert t["valid"] and "start at" in t["message"]
    assert "error" in s.execute("submit_job", {"script": "echo no shebang"})


def test_actions_need_confirmation_and_cancel_only_own_jobs():
    s, fake = live({"sbatch": (0, "1\n", "")}, confirm=lambda n, a: False)
    assert s.execute("submit_job", {"script": "#!/bin/bash\n"})["error"] == "the user declined this action"
    assert not any("sbatch" in " ".join(c[0]) for c in fake.calls)
    asked = []
    s, fake = live({"squeue&--me": (0, "", ""), "squeue&-j&5": (0, "", ""), "sacct": (0, "5|x|COMPLETED|0:0|p|00:01:00|"
                    "01:00:00|n1|||1|\n", "")}, confirm=lambda n, a: asked.append(n) or True)
    r = s.execute("cancel_job", {"job_id": "5"})  # finished: refused before the user is asked
    assert r["error"] == "job 5 is not one of your queued or running jobs (state: COMPLETED)" and not asked
    assert not any(c[0][0] == "scancel" for c in fake.calls)
    s, _ = live({"squeue&--me&-j&5": (0, "PENDING\n", "")}, confirm=lambda n, a: "no terminal to ask")
    assert s.execute("cancel_job", {"job_id": "5"}) == {"error": "no terminal to ask", "action": "cancel_job"}
    s, _ = live({"squeue&--me&%i": (0, "5\n", ""), "squeue&-j&5": (0, "5|x|PENDING|Priority|p|1|0:00|1:00||N/A\n", ""),
                 "scancel": (0, "", "")})
    assert s.execute("cancel_job", {"job_id": "5"}) == {"cancelled": True, "message": "Job 5 cancelled (was PENDING)"}


def test_read_job_log_uses_the_jobs_own_paths():
    s, fake = live({"scontrol&show&job": (0, "JobId=8 StdErr=/w/slurm-8.out StdIn=/dev/null StdOut=/w/slurm-8.out\n", ""),
                    "tail": (0, "line1\nline2\n", "")})
    r = s.execute("read_job_log", {"job_id": "8", "stream": "stderr", "tail_lines": 2})
    assert r["file"] == "/w/slurm-8.out" and r["lines"] == ["line1", "line2"] and "same file" in r["note"]
    assert ["tail", "-n", "2", "/w/slurm-8.out"] in [c[0] for c in fake.calls]
    with pytest.raises(ToolError):
        s.execute("read_job_log", {"job_id": "8", "tail_lines": 500})


def test_read_job_log_of_a_finished_job_expands_the_sacct_pattern():
    # As on Leonardo (Slurm 25.05): --output=slurm/logs/%x-%j.out and no --error, so sacct's StdErr is empty.
    sacct = "/w/repo/slurm/logs/%x-%j.out||/w/repo|dk-export|a08trc06|60340010\n"
    s, fake = live({"scontrol&show&job": (1, "", "slurm_load_jobs error: Invalid job id specified"),
                    "sacct&StdOut": (0, sacct, ""), "tail": (0, "done\n", "")})
    r = s.execute("read_job_log", {"job_id": "60340010", "stream": "stderr", "tail_lines": 1})
    assert r["file"] == "/w/repo/slurm/logs/dk-export-60340010.out" and r["lines"] == ["done"]
    assert "same file" in r["note"]


def test_read_job_log_patterns_for_array_tasks_relative_paths_and_defaults():
    row = {"JobName": "sweep", "User": "u1", "JobIDRaw": "4249"}
    assert slurm_live._expand("logs/%x_%A_%a_%j_%u_%%.out", "4242_7", row) == "logs/sweep_4242_7_4249_u1_%.out"
    assert slurm_live._expand("out-%5j.log", "42", {}) == "out-42.log"
    s, _ = live({"scontrol&show&job": (1, "", ""), "sacct&StdOut": (0, "logs/%j.out|logs/%j.err|/w|j|u|42\n", "")})
    assert s._log_path("42", "stderr") == "/w/logs/42.err"
    s, _ = live({"scontrol&show&job": (1, "", ""), "sacct&StdOut": (0, "||/w|j|u|42\n", "")})
    assert s._log_path("42", "stdout") == "/w/slurm-42.out"


def test_gpu_availability_counts_free_gpus_on_usable_nodes():
    out = ("lrdn0001 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:4(IDX:0-3) mixed\n"
           "lrdn0002 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:1(IDX:0) mixed\n"
           "lrdn0003 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:0(IDX:N/A) idle\n"
           "lrdn0004 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:0(IDX:N/A) drained\n")
    s, _ = live({"sinfo": (0, out, "")})
    assert s.execute("gpu_availability", {}) == {"partitions": [
        {"partition": "boost_usr_prod", "gpu_type": "a100", "gpus_total": 16, "gpus_free": 7, "idle_nodes": 1}],
        "free_gpus_by_type": {"a100": 7}}
    viz = out + "lrdn5001 lrd_all_viz gpu:quadro:2(S:0) gpu:quadro:0(IDX:N/A) idle\n"
    s, _ = live({"sinfo": (0, viz, "")})
    assert s.execute("gpu_availability", {})["free_gpus_by_type"] == {"a100": 7, "quadro": 2}


def test_invalid_calls():
    s, _ = live({})
    with pytest.raises(ToolError):
        s.execute("rm_rf", {})
    with pytest.raises(ToolError):
        s.execute("job_status", {})
    assert "invalid job id" in s.execute("job_status", {"job_id": "1; rm -rf ~"})["error"]


def test_job_usage_argv_rows_and_error_for_non_running_jobs():
    s, fake = live({"sstat&MaxVMSize": (0, "9|01:30:00|3145728K|1048576K|4194304K\n9.ext|01:00:00|||2097152K\n", "")})
    r = s.execute("job_usage", {"job_id": "9"})
    assert fake.calls[0][0] == ["sstat", "--allsteps", "-j", "9", "--format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize", "-P", "-n"]
    assert r["steps"][0] == {"JobID": "9", "AveCPU": "01:30:00", "MaxRSS": "3.0G", "AveRSS": "1.0G",
                             "MaxVMSize": "4.0G"} and r["steps"][1]["JobID"] == "9.ext"
    s, _ = live({"sstat": (1, "", "error: Pending job")})
    r = s.execute("job_usage", {"job_id": "9"})
    assert "not running" in r["error"] and "job_accounting" in r["error"]
    s, _ = live({"sstat": (0, "9.extern|9-00:00:00||\n9.batch|06:02:55|133615688K|1|2\n", "")})  # Curiosity's output
    assert [x["JobID"] for x in s.execute("job_usage", {"job_id": "9"})["steps"]] == ["9.batch"]


def test_job_script_of_a_finished_job_explains_why_it_is_missing():
    gone = (1, "", "job script retrieval failed: Invalid job id specified")
    s, _ = live({"scontrol&write": gone, "sacct&--batch-script": (0, "NONE\n", "")})
    assert "does not keep the batch scripts" in s.execute("job_script", {"job_id": "5"})["error"]
    s, _ = live({"scontrol&write": gone, "sacct&--batch-script": (0, "", "")})
    assert s.execute("job_script", {"job_id": "5"})["error"] == "no job 5 found"


def test_job_priority_sprio_and_sshare_outputs_with_exit_codes():
    s, fake = live({"sprio&-j&9": (0, "JOBID PARTITION PRIORITY AGE FAIRSHARE\n9 p 100 90 10\n", ""),
                    "sshare&-U&-l": (0, "ACCOUNT USER SHARE LEVELS\nuser 100\n", "")})
    r = s.execute("job_priority", {"job_id": "9"})
    assert [c[0] for c in fake.calls] == [["sprio", "-j", "9", "-l"], ["sshare", "-U", "-l"]]
    assert "9 p 100" in r["priority"] and r["sprio_exit_code"] == 0 and "user 100" in r["shares"]
    assert "note" not in r
    s, _ = live({"sprio&-j&9": (0, "JOBID PARTITION PRIORITY AGE FAIRSHARE\n", "")})
    r = s.execute("job_priority", {"job_id": "9"})
    assert "not pending" in r["note"]


def test_update_job_validates_the_value_and_uses_scontrol_update():
    s, fake = live({"squeue&--me&-j": (0, "9\n", ""), "scontrol&update": (0, "", "")})
    assert s.execute("update_job", {"job_id": "9", "field": "TimeLimit", "value": "2:00:00"}) == {
        "updated": True, "job_id": "9", "field": "TimeLimit", "value": "2:00:00"}
    assert fake.calls[-1][0] == ["scontrol", "update", "JobId=9", "TimeLimit=2:00:00"]
    assert "invalid value" in s.execute("update_job", {"job_id": "9", "field": "TimeLimit", "value": "value 2"})["error"]


def test_hold_release_and_requeue_use_scontrol_on_own_jobs():
    s, fake = live({"squeue&--me&-j": (0, "9\n", ""), "scontrol&hold": (0, "", ""),
                    "scontrol&release": (0, "", ""), "scontrol&requeue": (0, "", "")})
    assert s.execute("hold_job", {"job_id": "9"}) == {"hold": True, "job_id": "9", "message": "Job 9 held"}
    assert s.execute("release_job", {"job_id": "9"}) == {"release": True, "job_id": "9", "message": "Job 9 released"}
    assert s.execute("requeue_job", {"job_id": "9"}) == {"requeue": True, "job_id": "9", "message": "Job 9 requeued"}
    arc = [[c[0][0], c[0][1]] for c in fake.calls if c[0][0] == "scontrol"]
    assert ["scontrol", "hold"] in arc and ["scontrol", "release"] in arc and ["scontrol", "requeue"] in arc
    s, _ = live({"squeue&--me": (0, "", "")})  # not one of the user's own jobs: refused before any scontrol
    s2fake = s._runner
    r = s.execute("hold_job", {"job_id": "9"})
    assert r == {"error": "job 9 is not one of your queued or running jobs (not found)", "action": "hold_job"}
    assert not any(c[0][0] == "scontrol" for c in s2fake.calls)


@pytest.mark.parametrize(("command", "ok"), [("squeue --me", True), ("scontrol show job 123", True),
                                             ("rm -rf /", False), ("scontrol update JobId=1 TimeLimit=1", False),
                                             ("squeue | grep x", False), ("sinfo; ls", False)])
def test_run_readonly_command_allowlist(command, ok):
    s, fake = live({"squeue&--me": (0, "1 job\n", ""), "scontrol&show": (0, "JobId=123\n", "")})
    r = s.execute("run_readonly_command", {"command": command})
    if ok:
        assert "error" not in r and r["exit_code"] == 0 and r["command"] == command
        assert fake.calls[-1][0] == shlex.split(command)
    else:
        assert "refused" in r["error"]


def test_run_readonly_command_truncates_long_output():
    s, _ = live({"squeue": (0, "x" * 9000 + "\n", "")})
    r = s.execute("run_readonly_command", {"command": "squeue --me"})
    assert len(r["stdout"]) == 8011 and r["stdout"].endswith("[truncated]")


def test_new_actions_call_confirm_and_return_the_decline_error():
    for name, args in [("update_job", {"job_id": "9", "field": "TimeLimit", "value": "2:00:00"}),
                       ("hold_job", {"job_id": "9"}), ("release_job", {"job_id": "9"}),
                       ("requeue_job", {"job_id": "9"})]:
        s, fake = live({"squeue&--me&-j&9": (0, "PENDING\n", "")}, confirm=lambda n, a: False)
        r = s.execute(name, args)
        assert r["error"] == "the user declined this action" and r["action"] == name
        assert all(c[0][0] == "squeue" for c in fake.calls)  # only the check that the job is still queued


def test_round_limit_appends_a_stop_notice_with_the_last_call(monkeypatch, tmp_path):
    s, _ = live({})
    bot = assistant.Assistant(tmp_path, docs=False, slurm=s)
    def stream(messages):  # always wants another tool round, never a final answer
        yield {'tool_calls': [{'name': 'hold_job', 'arguments': '{"job_id": "9"}'}]}
    monkeypatch.setattr(bot, "_stream", stream)
    out = StringIO()
    answer, _ = bot.ask("hold job 9", out=out, max_tool_rounds=1)
    assert "stopped after 1 tool rounds" in out.getvalue() and "hold_job" in out.getvalue()
    notice = bot.history[-1]
    assert notice["role"] == "assistant" and "hold_job" in notice["content"] and '"job_id": "9"' in notice["content"]


def test_agent_loop_runs_slurm_tools_and_returns_results(monkeypatch, tmp_path):
    s, _ = live({"squeue&--me": (0, "77|train|RUNNING|p|0:10|1:00:00|None\n", "")})
    bot = assistant.Assistant(tmp_path, docs=False, slurm=s)
    turns = iter([[{"tool_calls": [{"name": "list_queue", "arguments": "{}"}]}], ["You have one running job, 77."]])
    seen = []
    monkeypatch.setattr(bot, "_stream", lambda messages: (seen.append(list(messages)), next(turns))[1])
    answer, _ = bot.ask("what is running?", out=StringIO())
    assert answer == "You have one running job, 77."
    tool_msg = seen[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["jobs"][0]["job_id"] == "77"


def test_openai_history_links_tool_calls_and_results(monkeypatch, tmp_path):
    """vLLM rejects (HTTP 400) a follow-up request whose tool calls lack ids or whose results lack tool_call_id."""
    s, _ = live({"squeue&--me": (0, "", ""), "sinfo": (0, "", "")})
    bot = assistant.Assistant(tmp_path, docs=False, slurm=s, openai="http://127.0.0.1:1/v1")
    turns = iter([[{"tool_calls": [{"id": "chatcmpl-tool-1", "name": "list_queue", "arguments": "{}"},
                                   {"name": "gpu_availability", "arguments": "{}"}]}], ["done"]])
    seen = []
    monkeypatch.setattr(bot, "_stream", lambda messages: (seen.append(json.loads(json.dumps(messages))), next(turns))[1])
    bot.ask("queue and gpus?", out=StringIO())
    call_msg, r1, r2 = seen[1][-3:]
    ids = [c["id"] for c in call_msg["tool_calls"]]
    assert ids[0] == "chatcmpl-tool-1" and ids[1] and len(set(ids)) == 2
    assert all(isinstance(c["function"]["arguments"], str) for c in call_msg["tool_calls"])
    assert [r1["tool_call_id"], r2["tool_call_id"]] == ids and r1["role"] == r2["role"] == "tool"


def test_ssh_reuses_one_connection_unless_the_host_has_its_own(monkeypatch):
    class P:
        def __init__(self, out):
            self.stdout, self.stderr, self.returncode = out, "", 0
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if cmd[:2] == ["ssh", "-G"]:
            return P("user me\ncontrolpath /home/me/.ssh/cm-x\n" if cmd[2] == "own" else "user me\ncontrolmaster false\n")
        return P("me\n")
    monkeypatch.setattr(slurm_live.subprocess, "run", fake_run)
    monkeypatch.setattr(slurm_live.os, "name", "posix")  # connection sharing is off on Windows (tested below)
    SlurmLive(ssh="plain").run(["id", "-un"])
    assert "ControlPath=~/.ssh/clusterpilot-%C" in seen[-1] and seen[-1][-2:] == ["plain", "id -un"]
    SlurmLive(ssh="own").run(["id", "-un"])
    assert not any("Control" in c for c in seen[-1])


def test_guessed_partition_names_are_resolved_or_listed():
    sinfo = (0, "lrdn0001  boost_usr_prod  gpu:a100:4(S:0-1)  gpu:a100:2(IDX:0-1)  mix\n", "")
    s, fake = live({"sinfo&%R": (0, "boost_usr_prod\ndcgp_usr_prod\nlrd_all_serial\nboost_usr_prod\n", ""),
                    "sinfo&-N": sinfo, "squeue&--me": (0, "", "")})
    r = s.execute("gpu_availability", {"partition": "boost"})
    assert r["partitions"][0]["partition"] == "boost_usr_prod" and "used 'boost_usr_prod'" in r["note"]
    assert "-p" in fake.calls[-1][0] and "boost_usr_prod" in fake.calls[-1][0]
    r = s.execute("gpu_availability", {"partition": "leonardo"})
    assert "partitions" in r and "-p" not in fake.calls[-1][0] and "lrd_all_serial" in r["note"]
    r = s.execute("list_queue", {"partition": "booster"})
    assert r == {"error": "there is no partition 'booster'; partitions on this cluster: "
                          "boost_usr_prod, dcgp_usr_prod, lrd_all_serial"}
    assert s.execute("list_queue", {"partition": "lrd_all_serial"}) == {"jobs": []}


def test_web_page_approves_agent_actions_with_buttons(monkeypatch):
    import re
    import threading
    import time
    from io import BytesIO
    from distillkit import assistant_web
    from distillkit.assistant_web import make_handler

    monkeypatch.setattr(assistant_web, "APPROVAL_WAIT", 5)

    class Bot:  # stands in for Assistant: one cancel_job call per question
        model, builtin, history = "m", False, [{}]

        def __init__(self, slurm):
            self.slurm = slurm

        def ask(self, question, out):
            out.write(json.dumps(self.slurm.execute("cancel_job", {"job_id": "5"})))

        safe_ask = ask

    s, fake = live({"squeue&--me": (0, "5\n", ""), "squeue&-j&5": (0, "5|x|PENDING|Priority|p|1|0:00|1:00||N/A\n", ""),
                    "scancel": (0, "", "")})
    handler = make_handler(Bot(s), approve_in_page=True)

    class Fake(handler):
        def __init__(self, path, body):
            self.path, self.rfile, self.wfile = path, BytesIO(body), BytesIO()
            self.headers = {"Content-Length": str(len(body))}
            self.request_version, self.requestline, self.command = "HTTP/1.1", f"POST {path} HTTP/1.1", "POST"

    for approve, expect in ((True, b'"cancelled": true'), (False, b"did not approve")):
        ask = Fake("/api/ask", b'{"question": "cancel 5"}')
        t = threading.Thread(target=ask.do_POST)
        t.start()
        for _ in range(200):
            if m := re.search(rb"\[\[approve (\w+) (.*)\]\]", ask.wfile.getvalue()):
                break
            time.sleep(0.01)
        assert b"cancel job 5" in m.group(2).lower()
        Fake("/api/approve", json.dumps({"id": m.group(1).decode(), "approve": approve}).encode()).do_POST()
        t.join(5)
        assert expect in ask.wfile.getvalue()
    assert sum(c[0][0] == "scancel" for c in fake.calls) == 1


def test_run_readonly_command_fills_in_the_user_name():
    s, fake = live({"id&-un": (0, "alice\n", ""), "sacctmgr&show": (0, "acct|alice\n", "")})
    r = s.execute("run_readonly_command", {"command": "sacctmgr show assoc user=$USER"})
    assert r["exit_code"] == 0 and fake.calls[-1][0] == ["sacctmgr", "show", "assoc", "user=alice"]
    assert "refused" in s.execute("run_readonly_command", {"command": "squeue -u $HOME"})["error"]


@pytest.mark.parametrize(("command", "ok"), [("sacctmgr -P show assoc user=alice", True), ("scontrol -o show job 1", True),
                                             ("sacctmgr -i delete user alice", False), ("sacctmgr -M show", True),
                                             ("scontrol update show=1", False), ("man scontrol", False)])
def test_run_readonly_command_allows_options_before_show(command, ok):
    s, fake = live({"sacctmgr": (0, "x\n", ""), "scontrol": (0, "x\n", "")})
    r = s.execute("run_readonly_command", {"command": command})
    assert ("error" not in r) == ok
    if not ok:
        assert "search_docs" in r["error"]


def test_no_ssh_connection_sharing_on_windows(monkeypatch):
    monkeypatch.setattr(slurm_live.os, "name", "nt")  # Windows OpenSSH has no ControlMaster
    assert SlurmLive(ssh="host").mux_options() == []
