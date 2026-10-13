import json
from pathlib import Path
import shutil

import pytest

from distillkit import eval_v3
from distillkit.io import read_jsonl, write_jsonl

EVAL_ROOT = Path(__file__).resolve().parents[1] / "data/eval"


def test_audit_uses_module_checkout_despite_older_eval_in_working_directory(tmp_path, monkeypatch):
    other_checkout = tmp_path / "shared"
    write_jsonl(other_checkout / "data/eval/old.jsonl", [{"question": "Old evaluation only."}])
    monkeypatch.chdir(other_checkout)
    released = read_jsonl(EVAL_ROOT / "eval_v3.jsonl")
    rows = [
        {"id": "source-leak", "doc_id": "eval_v3_resources", "question": "A different question about source provenance."},
        {"id": "question-leak", "doc_id": "other", "question": "  " + released[0]["question"].upper() + "  "},
        {"id": "clean", "doc_id": "other", "question": "How do dolphins navigate in the ocean?"},
    ]
    write_jsonl(Path("b7/verified.jsonl"), rows)
    output = Path("audit-result")
    eval_v3.audit([Path("b7/verified.jsonl")], output)
    result = json.loads((output / "audit.json").read_text())[0]
    assert (result["rows"], result["source_overlap"], result["question_overlap"], result["excluded_union"]) == (3, 1, 1, 2)
    assert read_jsonl(output / "00-b7/clean.jsonl") == [rows[2]]
    assert read_jsonl(Path("b7/verified.jsonl")) == rows
    context = json.loads((output / "audit_context.json").read_text())
    assert Path(context["eval_root"]) == EVAL_ROOT
    assert context["question_files"]["eval_v3.jsonl"] == eval_v3.sha((EVAL_ROOT / "eval_v3.jsonl").read_bytes())
    assert context["eval_v3_sha256"] == eval_v3.check_release()["sha256"]


@pytest.mark.parametrize("manifest", [None, {"version": "eval-source-holdout-v1", "sources": []}])
def test_audit_fails_before_writing_if_required_source_holdout_is_missing_or_empty(tmp_path, monkeypatch, manifest):
    root = tmp_path / "data/eval"
    write_jsonl(root / "old.jsonl", [{"question": "Existing questions are not enough."}])
    if manifest is not None:
        (root / "eval_v3.holdout.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(eval_v3, "EVAL_ROOT", root)
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="holdout"):
        eval_v3.audit([], output)
    assert not output.exists()


def test_audit_requires_final_unchanged_questions_before_creating_outputs(tmp_path, monkeypatch):
    root = tmp_path / "data/eval"
    root.mkdir(parents=True)
    for name in ("eval_v3.holdout.json", "eval_v3.manifest.json", "eval_v3.jsonl"):
        shutil.copyfile(EVAL_ROOT / name, root / name)
    monkeypatch.setattr(eval_v3, "EVAL_ROOT", root)
    with (root / "eval_v3.jsonl").open("a") as f:
        f.write("\n")
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="release data changed"):
        eval_v3.audit([], output)
    assert not output.exists()


def test_audit_does_not_treat_pre_release_sources_as_a_final_question_audit(tmp_path, monkeypatch):
    root = tmp_path / "data/eval"
    root.mkdir(parents=True)
    shutil.copyfile(EVAL_ROOT / "eval_v3.holdout.json", root / "eval_v3.holdout.json")
    monkeypatch.setattr(eval_v3, "EVAL_ROOT", root)
    output = tmp_path / "audit"
    with pytest.raises(FileNotFoundError, match="eval_v3.manifest.json"):
        eval_v3.audit([], output)
    assert not output.exists()
