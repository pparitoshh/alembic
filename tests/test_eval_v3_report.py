import importlib.util
import json
from pathlib import Path

import pytest

from distillkit.io import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_v3_report", ROOT / "research/eval_v3/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def completed(tmp_path):
    rows = [{"id":"a", "doc_id":"source", "task":"howto", "question":"Question A", "reference":"Answer A"},
            {"id":"b", "doc_id":"source", "task":"debug", "question":"Question B", "reference":"Answer B"}]
    outputs = [{**r, "answer_base":"base", "answer_student":"student",
                "judge":{"student_vs_base": verdicts}} for r, verdicts in zip(rows, [["A","B"], ["T","A"]])]
    summary = {"judge_protocol":"paired-orders-v2-invalid-incomplete", "judge":"openai/gpt-oss-20b",
        "student_vs_base_win_rate": .625, "judge_comparisons":{"student_vs_base":{
            "status":"complete", "questions":2, "complete_pairs":2, "invalid_verdicts":0}}}
    write_jsonl(tmp_path / "eval_outputs.jsonl", outputs)
    (tmp_path / "eval_summary.json").write_text(json.dumps(summary))
    return tmp_path, rows, outputs, summary


def test_report_recomputes_equal_question_weighted_paired_score(completed):
    directory, rows, _, _ = completed
    _, scores = report.checked_result(directory, rows)
    assert scores == {"a":1, "b":.25}


def test_report_rejects_invalid_judging_even_if_summary_claims_complete(completed):
    directory, rows, outputs, _ = completed
    outputs[0]["judge"]["student_vs_base"][0] = None
    write_jsonl(directory / "eval_outputs.jsonl", outputs)
    with pytest.raises(ValueError, match="incomplete paired"):
        report.checked_result(directory, rows)


def test_report_rejects_different_release_and_incorrect_summary(completed):
    directory, rows, outputs, summary = completed
    rows[0]["reference"] = "Changed reference"
    with pytest.raises(ValueError, match="differs from release"):
        report.checked_result(directory, rows)
    rows[0]["reference"] = outputs[0]["reference"]
    summary["student_vs_base_win_rate"] = .9
    (directory / "eval_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="headline win rate"):
        report.checked_result(directory, rows)
