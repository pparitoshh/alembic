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
    assert parse_json(JudgeVerdict, '{"verdict": "A"}') is None


def test_question_too_short():
    assert parse_json(GeneratedQuestion, '{"question": "Why?"}') is None
    assert parse_json(GeneratedQuestion, '{"question": "Why is my job stuck in PD?"}') is not None
