"""research/train_multistep_tools/collect_sim.py: simulated Slurm world, tool outputs and the teacher loop (no network)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillkit.assistant import SYSTEM_PROMPT
from distillkit.slurm_live import LIVE_SCHEMAS, openai_tools

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("collect_sim", ROOT / "research/train_multistep_tools/collect_sim.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def task(fixture="failed_oom", cluster="curiosity", tid="t1", req="Why did job {JOB_ID} fail?"):
    return {"id": tid, "cluster": cluster, "family": "debug", "user_request": req, "expected_tools": ["job_status"],
            "needs_tool": True, "multi_step": True, "fixture": fixture}


def sim(fixture, cluster="curiosity", tid="t1"):
    w = cs.make_world(task(fixture, cluster, tid))
    return w, cs.SimSlurm(w)


def test_world_is_consistent_per_task_and_job_id_has_6_or_7_digits():
    a, b = cs.make_world(task(tid="x")), cs.make_world(task(tid="x"))
    assert a == b
    assert a["job_id"].isdigit() and len(a["job_id"]) in (6, 7)
    assert cs.make_world(task(tid="y"))["job_id"] != a["job_id"]


def test_simulator_covers_every_live_tool():
    assert {s["name"] for s in LIVE_SCHEMAS} == {t["function"]["name"] for t in openai_tools()}
    assert len(LIVE_SCHEMAS) == 18
    for s in LIVE_SCHEMAS:
        assert callable(getattr(cs.SimSlurm, s["name"]))


@pytest.mark.parametrize("cluster", ["curiosity", "leonardo"])
def test_every_tool_answers_with_valid_arguments(cluster):
    w, s = sim("running", cluster)
    jid = w["job_id"]
    calls = {"job_status": {"job_id": jid}, "list_queue": {}, "submit_job": {"script": "#!/bin/bash\necho hi\n", "test_only": True},
             "cancel_job": {"job_id": jid}, "read_job_log": {"job_id": jid}, "job_accounting": {"job_id": jid},
             "gpu_availability": {}, "partition_info": {}, "recent_jobs": {}, "job_script": {"job_id": jid},
             "job_usage": {"job_id": jid}, "job_priority": {"job_id": jid},
             "update_job": {"job_id": jid, "field": "TimeLimit", "value": "00:30:00"}, "hold_job": {"job_id": jid},
             "release_job": {"job_id": jid}, "requeue_job": {"job_id": jid},
             "run_readonly_command": {"command": "squeue --me"}, "reservation_info": {}}
    assert set(calls) == {x["name"] for x in LIVE_SCHEMAS}
    for name, args in calls.items():
        res = s.execute(name, json.dumps(args))
        assert isinstance(res, dict) and json.dumps(res)
    assert len(s.log) == 18


def test_schema_validation_is_the_real_code():
    _, s = sim("running")
    with pytest.raises(cs.ToolError, match="missing required"):
        s.execute("job_status", "{}")
    with pytest.raises(cs.ToolError, match="unknown tool"):
        s.execute("nope", "{}")
    with pytest.raises(cs.ToolError, match="must be one of"):
        s.execute("list_queue", '{"state": "DONE"}')


def test_running_fixture():
    w, s = sim("running")
    st = s.execute("job_status", {"job_id": w["job_id"]})
    assert st["state"] == "RUNNING" and st["node_list"].startswith("curiosity-") and st["partition"] == "hackathon"
    assert s.execute("job_usage", {"job_id": w["job_id"]})["steps"][0]["JobID"] == w["job_id"] + ".batch"
    assert any(j["job_id"] == w["job_id"] for j in s.execute("list_queue", {})["jobs"])


def test_pending_fixture():
    w, s = sim("pending", "leonardo")
    st = s.execute("job_status", {"job_id": w["job_id"]})
    assert st["state"] == "PENDING" and st["reason"] and "est_start" in st
    assert "does not exist" in s.execute("read_job_log", {"job_id": w["job_id"]})["error"]
    assert w["job_id"] in s.execute("job_priority", {"job_id": w["job_id"]})["priority"]


def test_completed_fixture():
    w, s = sim("completed")
    a = s.execute("job_accounting", {"job_id": w["job_id"]})
    assert a["state"] == "COMPLETED" and a["exit_code"] == "0:0" and "max_rss" in a
    assert "no live usage" in s.execute("job_usage", {"job_id": w["job_id"]})["error"]


def test_failed_oom_fixture_both_kinds():
    kinds = set()
    for i in range(30):
        w, s = sim("failed_oom", tid=f"oom{i}")
        log = "\n".join(s.execute("read_job_log", {"job_id": w["job_id"], "tail_lines": 50})["lines"])
        st = s.execute("job_status", {"job_id": w["job_id"]})
        kind = w["jobs"][w["job_id"]]["kind"]
        kinds.add(kind)
        if kind == "cuda_oom":
            assert "CUDA out of memory" in log and st["state"] == "FAILED"
        else:
            assert "oom-kill" in log and st["state"] == "OUT_OF_MEMORY"
    assert kinds == {"cuda_oom", "host_oom"}


def test_failed_time_fixture():
    w, s = sim("failed_time")
    log = s.execute("read_job_log", {"job_id": w["job_id"]})["lines"]
    st = s.execute("job_status", {"job_id": w["job_id"]})
    assert "DUE TO TIME LIMIT" in log[-1] and st["state"] == "TIMEOUT" and st["elapsed"] == st["time_limit"]


def test_failed_error_fixture():
    w, s = sim("failed_error", "leonardo")
    log = s.execute("read_job_log", {"job_id": w["job_id"], "stream": "stdout"})["lines"]
    assert "Traceback (most recent call last):" in log
    assert w["jobs"][w["job_id"]]["error_line"] in log
    assert s.execute("job_status", {"job_id": w["job_id"]})["exit_code"] == "1:0"


def test_unknown_job_and_cluster_facts():
    w, s = sim("running")
    assert s.execute("job_status", {"job_id": "1"})["found"] is False
    p = s.execute("partition_info", {"partition": "hackathon"})
    assert (p["nodes"], p["gpus_per_node"], p["gpu_type"], p["cpus_per_node"]) == (16, 8, "b300", 256)
    g = s.execute("gpu_availability", {})
    assert g["your_limits"]["qos"] == "eai-team04" and g["your_limits"]["gpu_limit"] == 8
    assert s.execute("reservation_info", {}) == {"reservations": []}
    w, s = sim("running", "leonardo")
    names = {p["partition"] for p in s.execute("partition_info", {})["partitions"]}
    assert names == {"boost_usr_prod", "dcgp_usr_prod", "lrd_all_serial"}
    assert s.execute("partition_info", {"partition": "boost_usr_prod"})["max_time"] == "1-00:00:00"
    assert s.execute("reservation_info", {})["reservations"][0]["name"].startswith("s_tra_ncc")
    assert "partitions on this cluster" in s.execute("partition_info", {"partition": "gpuq"})["error"]


def test_actions_change_the_world():
    w, s = sim("pending", "leonardo")
    jid = w["job_id"]
    assert s.execute("hold_job", {"job_id": jid})["hold"] is True
    assert s.execute("job_status", {"job_id": jid})["reason"] == "JobHeldUser"
    assert s.execute("release_job", {"job_id": jid})["release"] is True
    assert s.execute("update_job", {"job_id": jid, "field": "TimeLimit", "value": "48:00:00"})["updated"] is False
    assert s.execute("update_job", {"job_id": jid, "field": "TimeLimit", "value": "02:00:00"})["updated"] is True
    assert s.execute("cancel_job", {"job_id": jid})["cancelled"] is True
    assert s.execute("job_status", {"job_id": jid})["state"] == "CANCELLED"
    assert "not one of your" in s.execute("cancel_job", {"job_id": jid})["error"]
    assert [a["action"] for a in w["actions"]] == ["hold_job", "release_job", "update_job", "cancel_job"]


def test_submit_job_test_only_and_real():
    w, s = sim("completed", "leonardo")
    script = ("#!/bin/bash\n#SBATCH -A tra26_ncc1\n#SBATCH -p boost_usr_prod\n#SBATCH --gres=gpu:4\n"
              "#SBATCH --time=02:00:00\nsrun python x.py\n")
    t = s.execute("submit_job", {"script": script, "test_only": True})
    assert t["valid"] and t["message"].startswith("sbatch: Job ") and "to start at" in t["message"]
    assert not w["submitted"]
    r = s.execute("submit_job", {"script": script})
    assert r["submitted"] and r["message"] == f"Submitted batch job {r['job_id']}"
    assert s.execute("job_status", {"job_id": r["job_id"]})["state"] == "PENDING"
    assert w["submitted"] == [r["job_id"]]
    assert "account" in s.execute("submit_job", {"script": "#!/bin/bash\nhostname\n"})["error"]
    too_long = script.replace("02:00:00", "2-00:00:00")
    assert "time limit" in s.execute("submit_job", {"script": too_long})["error"]
    assert "full text" in s.execute("submit_job", {"script": "run.sh"})["error"]


def test_run_readonly_command_uses_the_real_guard():
    w, s = sim("running")
    assert "refused" in s.execute("run_readonly_command", {"command": "rm -rf /"})["error"]
    out = s.execute("run_readonly_command", {"command": "squeue --me | grep " + w["job_id"]})
    assert w["job_id"] in out["stdout"]


def test_job_id_placeholder_and_no_job():
    w = cs.make_world(task("failed_time"))
    assert "{JOB_ID}" not in task()["user_request"].replace("{JOB_ID}", w["job_id"])
    w2 = cs.make_world(task(None, req="How many GPUs per node on Curiosity?"))
    assert w2["job_id"] is None


# ---- teacher loop with a fake OpenAI client ----
def call(name, args, cid):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeClient:
    """Plays a script of replies; each reply is a function of the messages so far."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        content, calls = self.script.pop(0)(kw["messages"])
        msg = SimpleNamespace(content=content, tool_calls=calls or None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def job_of(messages):
    return messages[1]["content"].split()[3]


def test_two_step_chain():
    client = FakeClient([
        lambda m: ("", [call("job_status", {"job_id": job_of(m)}, "c1")]),
        lambda m: ("", [call("read_job_log", {"job_id": job_of(m)}, "c2")]),
        lambda m: ("The job hit its time limit.", None),
    ])
    row = cs.run_task(client, "teacher", task("failed_time"))
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert row["messages"][0]["content"] == SYSTEM_PROMPT
    assert row["question"] == f"Why did job {row['world']['job_id']} fail?"
    assert "DUE TO TIME LIMIT" in row["messages"][5]["content"]
    assert [t["tool"] for t in row["tools"]] == ["job_status", "read_job_log"]
    assert row["pass"] is None and row["task"]["fixture"] == "failed_time" and row["n_tool_calls"] == 2
    assert row["world"]["jobs"][row["world"]["job_id"]]["state"] == "TIMEOUT"
    kw = client.requests[0]
    assert kw["tools"] == openai_tools() and kw["tool_choice"] == "auto"
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    json.dumps(row)


def test_six_call_cap():
    many = lambda m: ("", [call("list_queue", {}, f"c{len(m)}")])  # noqa: E731
    client = FakeClient([many] * 6 + [lambda m: ("Final answer.", None)])
    row = cs.run_task(client, "teacher", task("running"))
    assert row["n_tool_calls"] == 6 and len(row["tools"]) == 6
    assert client.requests[-1]["tool_choice"] == "none"
    assert row["messages"][-1] == {"role": "assistant", "content": "Final answer."} and row["error"] is None
    # one reply with more calls than the cap allows is cut; a model that still calls tools gets an error
    client = FakeClient([lambda m: ("", [call("list_queue", {}, f"x{i}") for i in range(8)]),
                         lambda m: ("", [call("list_queue", {}, "y")])])
    row = cs.run_task(client, "teacher", task("running"))
    assert row["n_tool_calls"] == 6 and row["capped"] and "more than 6" in row["error"]


def test_bad_tool_call_becomes_an_error_result():
    client = FakeClient([lambda m: ("", [call("job_status", {}, "c1")]), lambda m: ("Need a job id.", None)])
    row = cs.run_task(client, "teacher", task("running"))
    assert "missing required" in json.loads(row["messages"][3]["content"])["error"]


def test_collect_resume(tmp_path):
    out = tmp_path / "out.jsonl"
    tasks = [task("completed", tid=f"t{i}") for i in range(4)]
    answer = lambda m: ("Done.", None)  # noqa: E731
    assert cs.collect(FakeClient([answer] * 2), "m", tasks, out, concurrency=2, limit=2) == 2
    assert cs.collect(FakeClient([answer] * 2), "m", tasks, out, concurrency=2) == 2
    assert cs.collect(FakeClient([]), "m", tasks, out) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["t0", "t1", "t2", "t3"]
    assert set(rows[0]) >= {"id", "family", "question", "pass", "tools", "messages", "world", "task"}
