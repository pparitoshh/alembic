"""Stage `decontam`: exclude held-out source chunks and overlapping questions.

`verify` only compares against `eval.file`. This stage compares run_dir/verified.jsonl against the
questions of every JSONL file under data/eval/ (config eval set, tool slice, dataset_v1 development
and final), with the same word-3-gram Jaccard and threshold as verify (`verify.dedup_threshold`).
Writes run_dir/clean.jsonl, run_dir/contaminated.jsonl (with the matching held-out file and
question) and prints the counts. Source exclusions come from *.holdout.json manifests;
question checks are lexical, not a semantic leakage detector.
"""
import json
from pathlib import Path

from .config import Config
from .io import read_jsonl, write_jsonl
from .verify import _jaccard, _shingles
from .eval_holdout import load_source_holdouts, source_overlap

EVAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "eval"


def heldout_questions(root: Path = EVAL_ROOT) -> list[tuple[str, str, set]]:
    """(file, question, shingles) for every row with a question in root/**/*.jsonl."""
    out = []
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line).get("question")
            if isinstance(q, str) and q.strip():
                out.append((str(path.relative_to(root)), q, _shingles(q)))
    return out


def run(cfg: Config, root: Path = EVAL_ROOT) -> Path:
    held = heldout_questions(root)
    sources = load_source_holdouts(root)
    if not held:
        raise ValueError(f"no held-out questions found under {root}")
    clean, dirty = [], []
    for r in read_jsonl(cfg.run_dir / "verified.jsonl"):
        source_hit = source_overlap(r, sources)
        if source_hit:
            dirty.append({**r, "contaminated_with": source_hit})
            continue
        sh = _shingles(r["question"])
        hit = next(((f, q) for f, q, h in held if _jaccard(sh, h) >= cfg.verify.dedup_threshold), None)
        if hit:
            dirty.append({**r, "contaminated_with": {"file": hit[0], "question": hit[1]}})
        else:
            clean.append(r)
    write_jsonl(cfg.run_dir / "clean.jsonl", clean)
    write_jsonl(cfg.run_dir / "contaminated.jsonl", dirty)
    print(f"[decontam] {len(clean)} clean, {len(dirty)} dropped against {len(held)} held-out questions "
          f"in {len({f for f, _, _ in held})} files and {len(sources)} source exclusions -> {cfg.run_dir / 'clean.jsonl'}")
    return cfg.run_dir / "clean.jsonl"
