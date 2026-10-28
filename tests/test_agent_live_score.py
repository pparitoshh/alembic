import json

import importlib.util

ROOT = __import__("pathlib").Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("agent_live_score", ROOT / "scripts/agent_live_score.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def _run():
    tasks = [
        {"id": "js-01", "expected_tool": "job_status",
         "expected_args": {"job_id": "<RUNNING_JOB>"}},
        {"id": "lq-01", "expected_tool": "list_queue",
         "expected_args": {"partition": "boost_usr_prod"}},
        {"id": "sj-01", "expected_tool": "submit_job",
         "expected_args": {"test_only": True}},
    ]
    rows = [
        {"id": "js-01", "calls": [{"name": "job_status", "args": {"job_id": "62422679"}, "result": "{}"}],
         "seconds": 7.0},
        # right tool called last, wrong tool first, partition in another case
        {"id": "lq-01",
         "calls": [{"name": "run_readonly_command", "args": {"command": "squeue --me"}, "result": "{}"},
                   {"name": "list_queue", "args": {"partition": "BOOST_USR_PROD"}, "result": "{}"}],
         "seconds": 9.0},
        {"id": "sj-01", "calls": [], "seconds": 4.0},
    ]
    for t in tasks:  # same placeholder fill as the scorer's main()
        t["expected_args"] = {k: {"<RUNNING_JOB>": "62422679"}.get(v, v) if isinstance(v, str) else v
                              for k, v in t["expected_args"].items()}
    return tasks, rows


def test_score_run_covers_all_rules():
    scores, per_tool = sc.score_run(*_run())
    js, lq, sj = scores["js-01"], scores["lq-01"], scores["sj-01"]
    assert (js["first_tool_ok"], js["any_tool_ok"], js["args_ok"], js["zero_calls"]) == (True, True, True, False)
    assert js["n_calls"] == 1 and js["seconds"] == 7.0
    assert (lq["first_tool_ok"], lq["any_tool_ok"], lq["args_ok"], lq["zero_calls"]) == (False, True, True, False)
    assert (sj["first_tool_ok"], sj["any_tool_ok"], sj["args_ok"], sj["zero_calls"]) == (False, False, False, True)
    assert json.dumps(scores), "all scores are json-serializable"
    assert per_tool["job_status"] == {"n": 1, "first": 1, "any": 1, "args": 1}
    assert per_tool["list_queue"] == {"n": 1, "first": 0, "any": 1, "args": 1}
    assert per_tool["submit_job"] == {"n": 1, "first": 0, "any": 0, "args": 0}


def test_submit_job_requires_test_only_true():
    calls_test = [{"name": "submit_job", "args": {"script": "#!/bin/bash\nhostname", "test_only": True}}]
    calls_live = [{"name": "submit_job", "args": {"script": "#!/bin/bash\nhostname", "test_only": False}}]
    assert sc.args_ok("submit_job", {"test_only": True}, calls_test)
    assert not sc.args_ok("submit_job", {"test_only": True}, calls_live)
    assert not sc.args_ok("submit_job", {"test_only": True}, [{"name": "submit_job", "args": {"script": "#!"}}])


def test_readonly_command_only_needs_the_key():
    assert sc.args_ok("run_readonly_command", {"command": "sinfo"}, [{"name": "run_readonly_command",
                                                                     "args": {"command": "squeue -h -j 1"}}])
    assert not sc.args_ok("run_readonly_command", {"command": "sinfo"}, [{"name": "run_readonly_command", "args": {}}])
