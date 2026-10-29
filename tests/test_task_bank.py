"""Tests for the step-2 task bank (research/train_multistep_tools/task_bank.py).

Hermetic: no cluster, no network, tmp dirs only. The mentor requests for the overlap check arrive as a
function argument or from a small fake usecases.jsonl in tmp_path, never from data/mentor_usecases/.
"""
import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.train_multistep_tools.task_bank import (
    ACTION_TOOLS, CLUSTERS, IMPLICIT_SHARE, FAMILIES, READ_ONLY_TOOLS, ROW_KEYS, TOOL_NAMES,
    build_tasks, f_gpu_then_partition, f_single_gpu_availability, is_implicit,
    load_mentor_requests, main, max_overlap,
)

FIXTURES = {None, "running", "pending", "completed", "failed_oom", "failed_time", "failed_error"}
OTHER_CLUSTER_FACTS = {"curiosity": ("hackathon", "B300", "eai-team04", "enroot", "pyxis"),
                       "leonardo": ("boost_usr_prod", "dcgp_usr_prod", "lrd_all_serial", "A100")}


def bank(n=6, seed=4, mentor=()):
    return build_tasks(seed, n, list(mentor))


def test_schema_and_tool_names():
    tasks, _ = bank(n=6)
    assert len(TOOL_NAMES) == 19
    assert {f"single_{t}" for t in READ_ONLY_TOOLS} == {f for f in FAMILIES if f.startswith("single_")}
    ids = set()
    for row in tasks:
        assert set(row) == ROW_KEYS
        assert re.fullmatch(r"tb-\d+-\d{5}", row["id"])
        assert row["cluster"] in ("curiosity", "leonardo")
        assert row["family"] in FAMILIES
        assert isinstance(row["expected_tools"], list)
        assert all(isinstance(t, str) and t in TOOL_NAMES for t in row["expected_tools"])
        assert row["needs_tool"] is bool(row["expected_tools"])
        assert row["multi_step"] is (len(row["expected_tools"]) >= 2)
        assert row["fixture"] in FIXTURES
        assert isinstance(row["user_request"], str) and row["user_request"]
        ids.add(row["id"])
    assert len(ids) == len(tasks)
    # no invented job ids: a request never contains a long number, only the {JOB_ID} placeholder
    assert not any(re.search(r"\b\d{5,}\b", row["user_request"]) for row in tasks)
    # action tools only where the user explicitly asks for the action
    for row in tasks:
        for tool in set(row["expected_tools"]) & ACTION_TOOLS:
            assert row["family"] in ("status_then_hold", "status_then_update")
    # every multi-step family really expects at least two tools
    for name, (expected, _) in FAMILIES.items():
        assert (name in MULTI_STEP_NAMES) == (len(expected) >= 2)


MULTI_STEP_NAMES = {name for name, (expected, _) in FAMILIES.items() if len(expected) >= 2}


def test_cluster_facts_stay_on_their_cluster():
    tasks, _ = bank(n=6)
    for row in tasks:
        for fact in OTHER_CLUSTER_FACTS["leonardo" if row["cluster"] == "curiosity" else "curiosity"]:
            assert fact not in row["user_request"]


def test_dcgp_is_never_a_gpu_request():
    assert CLUSTERS["leonardo"]["gpu_partitions"] == ["boost_usr_prod"]  # dcgp_usr_prod is CPU-only
    rng = random.Random(0)
    for _ in range(300):  # direct generator scan, many draws
        request, _ = f_gpu_then_partition(rng, CLUSTERS["leonardo"], False)
        assert "dcgp_usr_prod" not in request
        request, _ = f_single_gpu_availability(rng, CLUSTERS["leonardo"])
        assert "dcgp_usr_prod" not in request
    tasks, _ = bank(n=8)  # whole-bank scan: no GPU count may pair with dcgp_usr_prod
    for row in tasks:
        gpu_request = "gpu_availability" in row["expected_tools"] or re.search(r"\d+-GPU", row["user_request"])
        if gpu_request:
            assert "dcgp_usr_prod" not in row["user_request"]


def test_implicit_share_of_multi_step_requests():
    tasks, _ = bank(n=8)
    multi = [row for row in tasks if row["multi_step"]]
    implicit = sum(is_implicit(row["user_request"]) for row in multi) / len(multi)
    assert implicit >= 0.7  # at most 30% spell the steps out


def test_is_implicit_detector():
    assert is_implicit("Why does job 4711 still wait?")
    assert is_implicit("Is there room for a 4-GPU 12 h run on hackathon today?")
    assert is_implicit("Keep job 4711 from starting until tomorrow, please.")
    assert not is_implicit("First show me my queue, then look up the state of job 4711.")
    assert not is_implicit("Check the state of job 4711 and pull its accounting record.")
    assert not is_implicit("List my recent jobs, then give me the accounting record for job 4711.")


def test_multi_step_share_at_least_half():
    tasks, _ = bank(n=8)
    share = sum(row["multi_step"] for row in tasks) / len(tasks)
    assert share >= 0.5


def test_overlap_math():
    request = "please check the state of my job and read its log tail"
    assert max_overlap(request, [request]) == 1.0
    assert max_overlap(request, ["a totally unrelated sentence about cooking pasta for dinner"]) == 0.0
    assert max_overlap("short", [request]) == 0.0  # fewer than five words: no 5-grams, no overlap


def test_overlap_filter_drops_copy_keeps_other():
    base, _ = bank(n=6)
    copied = base[0]["user_request"]
    tasks, stats = bank(n=6, mentor=[copied])
    assert max_overlap(copied, [copied]) == 1.0
    requests = {row["user_request"] for row in tasks}
    assert copied not in requests  # the copy is dropped ...
    assert len(tasks) == len(base)  # ... and a different request takes its place
    assert stats["dropped_overlap"] >= 1
    assert all(max_overlap(row["user_request"], [copied]) <= 0.3 for row in tasks)


def test_no_duplicate_requests():
    tasks, _ = bank(n=6)
    requests = [row["user_request"] for row in tasks]
    assert len(requests) == len(set(requests))


def test_determinism_same_seed():
    mentor = ["run a two node torchrun training and report the outcome to me afterwards"]
    tasks_a, stats_a = build_tasks(4, 5, mentor)
    tasks_b, stats_b = build_tasks(4, 5, mentor)
    assert json.dumps(tasks_a, sort_keys=True) == json.dumps(tasks_b, sort_keys=True)
    assert stats_a == stats_b
    tasks_c, _ = build_tasks(9, 5, mentor)
    assert tasks_c != tasks_a  # a different seed gives a different bank


def test_job_id_placeholder_iff_fixture_set():
    tasks, _ = bank(n=10)
    for row in tasks:
        assert ("{JOB_ID}" in row["user_request"]) == (row["fixture"] is not None)
    assert any(row["fixture"] is None for row in tasks)
    assert any(row["fixture"] is not None for row in tasks)


def test_fake_mentor_file_and_cli(tmp_path):
    mentor_dir = tmp_path / "mentor" / "fakecluster"
    mentor_dir.mkdir(parents=True)
    (mentor_dir / "usecases.jsonl").write_text(
        json.dumps({"user_request": "serve a model with vLLM inside a container on the cluster"}) + "\n",
        encoding="utf-8")
    out = tmp_path / "tasks.jsonl"
    main([str(out), "--seed", "4", "--n-per-family", "2", "--mentor-dir", str(tmp_path / "mentor")])
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows and all(set(row) == ROW_KEYS for row in rows)
    assert [row["id"] for row in rows] == [f"tb-4-{i:05d}" for i in range(len(rows))]


def test_load_mentor_requests_reads_fake_file(tmp_path):
    mentor_dir = tmp_path / "mentor" / "x"
    mentor_dir.mkdir(parents=True)
    (mentor_dir / "usecases.jsonl").write_text(
        json.dumps({"user_request": "first fake request"}) + "\n"
        + json.dumps({"user_request": "second fake request"}) + "\n", encoding="utf-8")
    assert load_mentor_requests(tmp_path / "mentor") == ["first fake request", "second fake request"]


def test_clusters_defined():
    assert set(CLUSTERS) == {"curiosity", "leonardo"}


@pytest.mark.parametrize("tool", sorted(ACTION_TOOLS - {"hold_job", "update_job"}))
def test_submit_cancel_release_requeue_never_expected(tool):
    tasks, _ = bank(n=6)
    assert all(tool not in row["expected_tools"] for row in tasks)


def test_check_first_families_round2():
    for name in ("check_first_partition_info", "check_first_gpu_availability", "check_first_job_status"):
        expected, _ = FAMILIES[name]
        assert expected == [name[len("check_first_"):]] and name[len("check_first_"):] in READ_ONLY_TOOLS
    tasks, _ = bank(n=10)
    rows = [row for row in tasks if row["family"].startswith("check_first_")]
    assert rows
    for row in rows:
        assert row["needs_tool"] and not row["multi_step"]  # one read tool, so not counted as multi-step
        assert row["expected_tools"] == [row["family"][len("check_first_"):]]
        assert is_implicit(row["user_request"])  # goal-only wording: no tool words, no steps
        if row["family"] == "check_first_job_status":
            assert row["fixture"] is not None and "{JOB_ID}" in row["user_request"]
        else:
            assert row["fixture"] is None and "{JOB_ID}" not in row["user_request"]


def test_three_call_families_round2():
    assert FAMILIES["status_then_log_then_accounting"][0] == ["job_status", "read_job_log", "job_accounting"]
    assert FAMILIES["gpu_then_partition_then_reservation"][0] == [
        "gpu_availability", "partition_info", "reservation_info"]
    tasks, _ = bank(n=10)
    rows = [row for row in tasks if len(row["expected_tools"]) == 3]
    assert rows and all(row["multi_step"] for row in rows)


def test_implicit_share_still_seventy_percent_with_v3_families():
    assert IMPLICIT_SHARE == 0.7
    tasks, _ = bank(n=21)  # > 2 rows per multi family: the implicit quota and the explicit tail both sit in
    multi = [row for row in tasks if row["multi_step"]]
    implicit = sum(is_implicit(row["user_request"]) for row in multi) / len(multi)
    assert implicit >= 0.7
