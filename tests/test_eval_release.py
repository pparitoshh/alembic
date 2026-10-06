import hashlib
import json
from pathlib import Path
import unicodedata

import pytest
import yaml

from distillkit import verify
from distillkit.config import Config
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import prose_row

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "data/eval"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def test_release_is_exact_original_slices_plus_bound_supplement():
    manifest = json.loads((EVAL / "eval_all_v2.manifest.json").read_text())
    raw = (EVAL / "eval_all_v2.jsonl").read_bytes()
    originals = b""
    for part in manifest["composition"]:
        data = (ROOT / part["path"]).read_bytes()
        assert sha(data) == part["sha256"]
        assert len(data.splitlines()) == part["rows"]
        originals += data
    assert (EVAL / "eval_all.jsonl").read_bytes() == originals
    assert raw.startswith(originals)
    assert sha(raw[len(originals):]) == manifest["supplement"]["sha256"]
    assert sha(raw) == manifest["sha256"]
    assert len(raw.splitlines()) == manifest["rows"] == 66
    for doc, expected in manifest["held_out_sources"].items():
        assert sha((ROOT / "data/seeds" / f"{doc}.md").read_bytes()) == expected


def test_release_schema_ids_and_normalized_questions():
    rows = read_jsonl(EVAL / "eval_all_v2.jsonl")
    assert len({r["id"] for r in rows}) == len(rows)
    normalized = [" ".join(unicodedata.normalize("NFKC", r["question"]).casefold().split()) for r in rows]
    assert len(set(normalized)) == len(rows)
    for r in rows:
        assert set(r) == {"id", "doc_id", "task", "question", "reference"}
        assert all(isinstance(v, str) and v.strip() for v in r.values())
        assert r["doc_id"] in {"slurm_job_arrays", "slurm_requeue_signals"}
        assert r["task"] in {"concept", "howto", "debug", "script"}
    assert sum(r["id"] == "req-016" for r in rows) == 1
    excluded = {r["id"] for name in ("eval_tools.jsonl", "judge_calibration.jsonl") for r in read_jsonl(EVAL / name)}
    assert not excluded & {r["id"] for r in rows}


@pytest.mark.parametrize("name", ["qwen3_4b_qdora.yaml", "tools_pilot.yaml", "toy_qdora.yaml"])
def test_configs_use_release_without_changing_holdouts(name):
    raw = yaml.safe_load((ROOT / "configs" / name).read_text())
    cfg = Config.model_validate(raw)
    assert cfg.eval.file == Path("data/eval/eval_all_v2.jsonl")
    assert set(cfg.seeds.eval_docs) == {"slurm_job_arrays", "slurm_requeue_signals"}


def test_production_verifier_excludes_supplement_question(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/qwen3_4b_qdora.yaml").read_text())
    raw["run_dir"] = str(tmp_path / "run")
    raw["eval"]["file"] = str(ROOT / raw["eval"]["file"])
    cfg = Config.model_validate(raw)
    item = next(r for r in read_jsonl(cfg.eval.file) if r["id"] == "arr-cand-005")
    # Deliberate failing synthetic fixture, never a retained training example or reference answer.
    row = {"id": "synthetic-overlap", **prose_row({}, item["question"], "Synthetic placeholder answer.")}
    write_jsonl(cfg.run_dir / "generated.jsonl", [row])
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir / "verified.jsonl") == []
    assert read_jsonl(cfg.run_dir / "rejected.jsonl")[0]["reject_reason"] == "eval_contamination"
