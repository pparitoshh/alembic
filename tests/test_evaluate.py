from distillkit.evaluate import parse_verdict


def test_verdict_after_reasoning():
    assert parse_verdict("B") == "B"
    assert parse_verdict("Answer A invents --gpu-count, so B is better.\n\nB") == "B"
    assert parse_verdict("Verdict: t") == "T"


def test_verdict_missing():
    assert parse_verdict("") is None
    assert parse_verdict("Both are fine.") is None
