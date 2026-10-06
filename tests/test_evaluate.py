from distillkit.schemas import GeneratedQuestion, JudgeVerdict, parse_json

import json
from pathlib import Path
import pytest
from distillkit import evaluate
from distillkit.config import load_config
from distillkit.io import read_jsonl, write_jsonl


class SequenceJudge:
    def __init__(self, labels):
        self.labels = iter(labels)

    def map(self, fn, items):
        return [fn(item) for item in items]

    def chat_json(self, system, prompt, model, temperature=None):
        label = next(self.labels)
        return (model(verdict=label), json.dumps({"verdict": label})) if label else (None, "invalid response")


@pytest.mark.parametrize("labels,expected,bad", [
    (["A", "B"], 1.0, 0), (["B", "A"], 0.0, 0), (["T", "T"], 0.5, 0),
    (["A", "A"], 0.5, 0), ([None, "T"], None, 1), (["T", None], None, 1),
    ([None, None], None, 2),
])
def test_win_rate_preserves_invalid_verdicts(labels, expected, bad):
    rows = [{"question": "Why?", "reference": "Reference.", "answer_student": "s", "answer_base": "b"}]
    rate, invalid = evaluate.win_rate(SequenceJudge(labels), rows, "student", "base")
    assert rate == expected and invalid == bad
    assert rows[0]["judge"]["student_vs_base"] == labels
    assert len(rows[0]["judge_raw"]["student_vs_base"]) == 2


def test_invalid_pair_does_not_silently_shrink_the_denominator():
    rows = [{"question": f"Question {i}", "reference": "Reference.", "answer_student": "s", "answer_base": "b"} for i in range(2)]
    rate, invalid = evaluate.win_rate(SequenceJudge(["A", "B", None, "B"]), rows, "student", "base")
    assert rate is None and invalid == 1
    assert evaluate.win_rate(SequenceJudge([]), [], "student", "base") == (None, 0)


@pytest.mark.parametrize("model,tag", [("openai/gpt-oss-20b", ""), ("google/gemma-4-26B-A4B-it", "gemma4")])
def test_evaluate_writes_incomplete_status_and_raw_invalids(tmp_path, monkeypatch, model, tag):
    root = Path(__file__).parent.parent
    eval_path = tmp_path / "eval.jsonl"
    write_jsonl(eval_path, [{"id": "q1", "doc_id": "test", "task": "concept", "question": "What happens?", "reference": "Expected."}])
    cfg = load_config(root / "configs/toy_qdora.yaml", [f"run_dir={tmp_path / 'run'}", f"eval.file={eval_path}", "eval.tool_file=null", f"eval.tag={tag}", f"judge.model={model}"])
    monkeypatch.setattr(evaluate, "generate_answers", lambda *args: {"base": ["b"], "student": ["s"]})
    monkeypatch.setattr(evaluate, "Teacher", lambda *args, **kwargs: SequenceJudge([None, "T"]))
    summary = evaluate.run(cfg)
    assert summary["student_vs_base_win_rate"] is None
    assert summary["judge_protocol"] == evaluate.JUDGE_PROTOCOL
    assert summary["judge_comparisons"]["student_vs_base"] == {
        "status": "incomplete", "questions": 1, "expected_verdicts": 2, "invalid_verdicts": 1, "complete_pairs": 0,
    }
    suffix = f"_{tag}" if tag else ""
    saved = read_jsonl(cfg.run_dir / f"eval_outputs{suffix}.jsonl")[0]
    assert saved["judge"]["student_vs_base"] == [None, "T"]
    assert saved["judge_raw"]["student_vs_base"][0] == "invalid response"
    assert json.loads((cfg.run_dir / f"eval_summary{suffix}.json").read_text()) == summary


@pytest.mark.parametrize("fault", ["null", "all_null", "missing_metric", "missing_summary", "mixed_protocol", "mixed_judge"])
def test_seed_aggregation_cannot_hide_incomplete_or_mixed_comparisons(tmp_path, fault):
    from distillkit.aggregate import aggregate

    complete = {"student_vs_base_win_rate": 0.8, "judge_protocol": evaluate.JUDGE_PROTOCOL, "judge": "fixed-judge"}
    other = dict(complete)
    if fault in ("null", "all_null"):
        other["student_vs_base_win_rate"] = None
        if fault == "all_null":
            complete["student_vs_base_win_rate"] = None
    elif fault == "missing_metric":
        del other["student_vs_base_win_rate"]
    elif fault == "mixed_protocol":
        del other["judge_protocol"]
    elif fault == "mixed_judge":
        other["judge"] = "different-judge"
    for seed, summary in (("1", complete), ("2", other)):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        if not (fault == "missing_summary" and seed == "2"):
            (directory / "eval_summary.json").write_text(json.dumps(summary))
    result = aggregate(tmp_path)
    metric = result["metrics"]["student_vs_base_win_rate"]
    assert metric["mean"] is None and metric["status"] == "incomplete"
    assert metric["expected_n"] == 2 and len(result["judge_protocols"]) == 2


def test_seed_aggregation_cli_reports_null_without_crashing(tmp_path, monkeypatch, capsys):
    from distillkit.aggregate import main

    directory = tmp_path / "seed_42"
    directory.mkdir()
    (directory / "eval_summary.json").write_text(json.dumps({"student_vs_base_win_rate": None, "judge_protocol": evaluate.JUDGE_PROTOCOL}))
    monkeypatch.setattr("sys.argv", ["aggregate", str(tmp_path)])
    main()
    assert "INCOMPLETE" in capsys.readouterr().out
    saved = json.loads((tmp_path / "eval_seeds.json").read_text())
    assert saved["metrics"]["student_vs_base_win_rate"]["mean"] is None


def test_verdict_valid():
    v = parse_json(JudgeVerdict, '{"reasoning": "A invents --gpu-count.", "verdict": "B"}')
    assert v is not None and v.verdict == "B"


def test_verdict_in_code_fence():
    v = parse_json(JudgeVerdict, '```json\n{"reasoning": "same", "verdict": "T"}\n```')
    assert v is not None and v.verdict == "T"


def test_verdict_invalid():
    assert parse_json(JudgeVerdict, "") is None
    assert parse_json(JudgeVerdict, "B") is None  # bare letter is no longer accepted
    assert parse_json(JudgeVerdict, '{"reasoning": "x", "verdict": "C"}') is None
    assert parse_json(JudgeVerdict, '{}') is None  # verdict is required


def test_question_too_short():
    assert parse_json(GeneratedQuestion, '{"question": "Why?"}') is None
    assert parse_json(GeneratedQuestion, '{"question": "Why is my job stuck in PD?"}') is not None

def test_verdict_only_accepts_all_labels():
    import json

    schema = JudgeVerdict.model_json_schema()
    assert set(schema["properties"]) == {"verdict"}
    assert schema["required"] == ["verdict"]

    for label in ("A", "B", "T"):
        result = parse_json(JudgeVerdict, json.dumps({"verdict": label}))
        assert result is not None
        assert result.verdict == label

def test_verdict_only_rejects_invalid_payloads():
    for raw in (
        "{}",
        '{"reasoning":"x"}',
        '{"verdict":"C"}',
        '{"verdict":null}',
        '{"verdict":"A"',
        "A",
    ):
        assert parse_json(JudgeVerdict, raw) is None, raw
