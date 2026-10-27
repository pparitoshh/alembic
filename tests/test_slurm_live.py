"""Live Slurm backend: command construction and output parsing against recorded Slurm output (no cluster needed)."""
import json
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


def test_live_schemas_are_the_trained_ones_plus_job_script():
    assert slurm_live.LIVE_SCHEMAS[:len(SCHEMAS)] == SCHEMAS
    assert [t["function"]["name"] for t in slurm_live.openai_tools()][-1] == "job_script"


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
    s, _ = live({"sacct": (0, "7|job|COMPLETED|0:0|p|01:00:00|02:00:00|n1||02:00:00|4|cpu=4,gres/gpu:a100=2,node=1\n"
                              "7.batch|batch|COMPLETED|0:0||01:00:00||n1|3145728K|02:00:00|4|\n", "")})
    a = s.execute("job_accounting", {"job_id": "7"})
    assert a["cpu_efficiency"] == "50%" and a["max_rss"] == "3.0G" and a["gpus"] == 2


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
    s, fake = live({"squeue&--me": (0, "", "")})
    assert s.execute("cancel_job", {"job_id": "5"})["cancelled"] is False
    assert not any(c[0][0] == "scancel" for c in fake.calls)
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
        {"partition": "boost_usr_prod", "gpu_type": "a100", "gpus_total": 16, "gpus_free": 7, "idle_nodes": 1}]}


def test_invalid_calls():
    s, _ = live({})
    with pytest.raises(ToolError):
        s.execute("rm_rf", {})
    with pytest.raises(ToolError):
        s.execute("job_status", {})
    assert "invalid job id" in s.execute("job_status", {"job_id": "1; rm -rf ~"})["error"]


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
