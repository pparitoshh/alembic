import copy
import json
from pathlib import Path

import pytest

from distillkit import heldout
from distillkit.config import load_config
from distillkit.eval_v3 import (PROTOCOL, SYSTEM, QualityReview, digest, normalized,
                                generation_jobs, packet, preflight, quality_verdict, select, sha)
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import prose_row

ROOT = Path(__file__).resolve().parents[1]


def test_build_has_exact_frozen_sources_and_512_normal_candidates(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config("configs/gen_eval_v3.yaml")
    plan = preflight(cfg)
    jobs = generation_jobs(cfg)
    assert len(jobs) == 512
    assert len({j["doc_id"] for j in jobs}) == 8
    assert len({j["id"] for j in jobs}) == 512
    assert sum(j["task"] == "script" and j["doc_id"] == "eval_v3_multi_constraint" for j in jobs) == 64
    assert sum(j["task"] == "debug" and j["doc_id"] == "eval_v3_debug" for j in jobs) == 64
    assert sum(t["target"] for t in plan["topics"]) == 200


@pytest.fixture
def review_case():
    row = {"id": "candidate", "task": "howto", **prose_row({}, "How do I submit a job?", "Use sbatch.")}
    source = "Use sbatch to submit a job."
    verdict = QualityReview(topic="resources", self_contained=True, sound_premises=True,
        assessable=True, reference_complete=True, uncertainty="", explanation="Supported, self-contained request.")
    report = {"id": row["id"], "protocol": PROTOCOL, "packet_sha256": digest(packet(row, source)),
        "prompt_sha256": sha(SYSTEM.encode()), "reviewer": "independent-reviewer",
        "finish_reason": "stop", "review": verdict.model_dump(), "raw_response": verdict.model_dump_json()}
    return row, source, report


def test_quality_requires_bound_independent_review(review_case):
    row, source, report = review_case
    assert quality_verdict(row, source, report, "teacher")
    assert quality_verdict(row, source, report, "independent-reviewer") is None
    assert quality_verdict(row, source + " changed", report, "teacher") is None
    changed = copy.deepcopy(row)
    changed["messages"][-1]["content"] = "A different answer"
    assert quality_verdict(changed, source, report, "teacher") is None
    assert quality_verdict(row, source, None, "teacher") is None


@pytest.mark.parametrize("uncertainty", ["none", " NONE ", ""])
def test_explicit_no_uncertainty_sentinel_is_not_a_rejection(review_case, uncertainty):
    row, source, report = review_case
    report["review"]["uncertainty"] = uncertainty
    report["raw_response"] = json.dumps(report["review"])
    assert quality_verdict(row, source, report, "teacher")


@pytest.mark.parametrize("uncertainty", ["uncertain", "none except missing context", "certain", "N/A"])
def test_uncertainty_is_not_silently_normalized_to_acceptance(review_case, uncertainty):
    row, source, report = review_case
    report["review"]["uncertainty"] = uncertainty
    report["raw_response"] = json.dumps(report["review"])
    assert quality_verdict(row, source, report, "teacher") is None


@pytest.mark.parametrize("change", [{"finish_reason":"length"}, {"protocol":"old"},
    {"unexpected_tool_calls":[{}]}, {"raw_response":"{}"}, {"reviewer":""}])
def test_bad_quality_transport_is_not_acceptance(review_case, change):
    row, source, report = review_case
    assert quality_verdict(row, source, report | change, "teacher") is None


@pytest.mark.parametrize("change", [{"self_contained": False}, {"sound_premises": False},
    {"reference_complete": False}, {"assessable": False}, {"uncertainty": "Not enough context"},
    {"topic": "out_of_scope"}])
def test_negative_review_cannot_release(review_case, change):
    row, source, report = review_case
    report["review"].update(change)
    report["raw_response"] = json.dumps(report["review"])
    assert quality_verdict(row, source, report, "teacher") is None


def test_selection_is_order_independent_normalized_dedup_and_reports_deficits():
    candidates = [({"id":"1", "task":"howto", "question":"How can I submit this job?"}, "resources"),
                  ({"id":"2", "task":"debug", "question":"HOW CAN I SUBMIT THIS JOB?"}, "resources"),
                  ({"id":"3", "task":"script", "question":"Explain memory per CPU limits."}, "resources")]
    args = ({"resources": 3, "dependencies": 1}, ["  explain Memory per CPU limits. "], .8)
    chosen, deficits = select(candidates, *args)
    assert select(list(reversed(candidates)), *args) == (chosen, deficits)
    assert len(chosen) == 1
    assert deficits == {"resources": 2, "dependencies": 1}
    assert normalized("ＳＢＡＴＣＨ  option") == "sbatch option"


def test_decontam_applies_sources_even_when_questions_do_not_match(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg = load_config("configs/gen_eval_v3.yaml", [f"run_dir={tmp_path / 'run'}"])
    root = tmp_path / "eval"
    root.mkdir()
    write_jsonl(root / "questions.jsonl", [{"question": "An entirely unrelated evaluation question."}])
    (root / "source.holdout.json").write_text(json.dumps({"version": "eval-source-holdout-v1",
        "sources": [{"doc_id": "held", "kind": "chunk", "chunk_id": "held#0", "source_sha256": "a"*64}]}))
    rows = [{"id": "exclude", "doc_id":"held", "chunk_id":"held#0", "source_kind":"chunk",
             "source_sha256":"a"*64, "question":"Novel wording does not remove source leakage."},
            {"id":"keep", "doc_id":"unseen", "question":"How do CPU requests work?"}]
    write_jsonl(cfg.run_dir / "verified.jsonl", rows)
    heldout.run(cfg, root)
    assert [r["id"] for r in read_jsonl(cfg.run_dir / "clean.jsonl")] == ["keep"]
    assert read_jsonl(cfg.run_dir / "contaminated.jsonl")[0]["contaminated_with"]["reason"] == "heldout_source"
    assert read_jsonl(cfg.run_dir / "verified.jsonl") == rows


def test_stale_verified_input_cannot_bypass_training_source_holdout(tmp_path, monkeypatch):
    from distillkit.train import admitted_training_rows
    monkeypatch.chdir(ROOT)
    cfg = load_config("configs/gen_eval_v3.yaml", [f"run_dir={tmp_path / 'run'}"])
    # Full source is excluded even without a question or a usable GPU runtime.
    write_jsonl(cfg.run_dir / "verified.jsonl", [{"id":"leak", "doc_id":"eval_v3_resources"}])
    with pytest.raises(ValueError, match="overlap frozen evaluation sources"):
        admitted_training_rows(cfg)
