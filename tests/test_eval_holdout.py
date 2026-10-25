import hashlib
import json

import pytest

from distillkit.eval_holdout import load_source_holdouts, source_overlap


@pytest.fixture
def held(tmp_path):
    text = "Frozen source chunk."
    item = {"kind": "chunk", "doc_id": "manual", "chunk_id": "manual#2",
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
    (tmp_path / "eval.holdout.json").write_text(json.dumps({
        "version": "eval-source-holdout-v1", "sources": [item]}))
    return load_source_holdouts(tmp_path)


def test_question_rewording_cannot_bypass_source_holdout(held):
    row = {"doc_id": "manual", "chunk_id": "manual#2", "source_kind": "chunk",
           "source_sha256": "a" * 64, "question": "Entirely different question"}
    assert source_overlap(row, held)["reason"] == "heldout_source"


def test_renamed_source_is_excluded_by_hash(held):
    assert source_overlap({"doc_id": "alias", "source_sha256": held[0]["source_sha256"]}, held)


def test_other_bound_chunk_stays_available(held):
    assert source_overlap({"doc_id": "manual", "chunk_id": "manual#3",
                           "source_kind": "chunk", "source_sha256": "b" * 64}, held) is None


@pytest.mark.parametrize("extra", [{}, {"source_kind": "full"},
                                     {"source_kind": "chunk", "chunk_id": "manual#3"}])
def test_full_or_unbound_document_is_conservatively_excluded(held, extra):
    assert source_overlap({"doc_id": "manual", **extra}, held)


def test_invalid_manifest_fails_closed(tmp_path):
    (tmp_path / "bad.holdout.json").write_text('{"version":"unknown"}')
    with pytest.raises(ValueError, match="version"):
        load_source_holdouts(tmp_path)


def test_holdout_text_is_hash_bound(tmp_path):
    (tmp_path / "bad.holdout.json").write_text(json.dumps({"version":"eval-source-holdout-v1",
        "sources":[{"kind":"chunk","doc_id":"d","chunk_id":"d#0",
                    "source_sha256":"a"*64,"text":"changed"}]}))
    with pytest.raises(ValueError, match="text/hash"):
        load_source_holdouts(tmp_path)
