import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from distillkit import support_v3
from distillkit.records import prose_row

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_v3_retry", ROOT / "research/eval_v3/retry_support.py")
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


def test_id_schema_uses_exact_visible_packet_ids_without_changing_verdicts():
    source = "The source states this."
    row = prose_row({"id":"synthetic", "source_sha256":hashlib.sha256(source.encode()).hexdigest()},
                    "What is stated?", "An explicit statement. Another statement.")
    packet = support_v3.evidence_packet(row, source)
    original = support_v3.SupportReview.model_json_schema()
    schema = retry.constrained_schema(original, packet)
    assert "enum" not in original["$defs"]["UnitReview"]["properties"]["unit_id"]
    item = schema["$defs"]["UnitReview"]["properties"]
    assert item["unit_id"]["enum"] == [u["unit_id"] for u in packet["units"]]
    assert set(item["evidence_ids"]["items"]["enum"]) == {
        eid for t in packet["turns"] for eid in t["available_evidence_ids"]}
    assert item["verdict"] == original["$defs"]["UnitReview"]["properties"]["verdict"]
    assert schema["properties"]["units"]["minItems"] == schema["properties"]["units"]["maxItems"] == len(packet["units"])


@pytest.fixture
def positive_invalid():
    return {"review":{"units":[{"verdict":"supported"}], "uncertainty":{"present":False}}}


def test_only_structural_failures_are_retryable(positive_invalid):
    for reason in retry.STRUCTURAL_ERRORS:
        assert retry.retryable(positive_invalid, {"status":"uncertain", "reason":reason})
    for result in [{"status":"supported"}, {"status":"unsupported"}, {"status":"uncertain"},
                   {"status":"uncertain", "reason":"stale or mismatched review binding"}]:
        assert not retry.retryable(positive_invalid, result)


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_negative_units_remain_excluded_even_when_ids_are_invalid(positive_invalid, verdict):
    report = copy.deepcopy(positive_invalid)
    report["review"]["units"].append({"verdict":verdict})
    assert not retry.retryable(report, {"status":"uncertain", "reason":"unknown or duplicate target unit"})


def test_global_uncertainty_is_not_rerolled(positive_invalid):
    positive_invalid["review"]["uncertainty"]["present"] = True
    assert not retry.retryable(positive_invalid, {"status":"uncertain", "reason":"supported unit lacks evidence"})
