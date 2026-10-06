from distillkit.schemas import GeneratedQuestion, JudgeVerdict, parse_json


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
