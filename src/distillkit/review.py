"""Stage `review`: an independent model checks every generated record against its source.

Runs the existing support review (`verify.grounding_review_protocol`, e.g. source-support-v3) on
each row of run_dir/generated.jsonl, using the `judge` endpoint as the reviewer (serve a model other
than the teacher, e.g. --set judge.model=google/gemma-4-26B-A4B-it). Reports are appended to
`verify.grounding_reviews` (default run_dir/grounding_reviews.jsonl); rows already reviewed are
skipped, so a killed job resumes. `verify` with `require_grounding_review: true` then rejects
unsupported answers and holds uncertain ones as pending. Rows whose source cannot be resolved are
not reviewed; verify holds them as pending.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Config
from .io import read_jsonl
from .support import request_review, source_for_record


def reviews_path(cfg: Config) -> Path:
    return cfg.verify.grounding_reviews or cfg.run_dir / "grounding_reviews.jsonl"


def run(cfg: Config, reviewer=None) -> Path:
    out = reviews_path(cfg)
    rows = read_jsonl(cfg.run_dir / "generated.jsonl")
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo, unresolved = [], 0
    resolver = None
    if cfg.seeds.registry is not None:
        from .source_resolver import SourceResolver
        resolver = SourceResolver(cfg)
    for r in rows:
        if r["id"] in done or r.get("doc_id") in cfg.seeds.eval_docs:
            continue
        try:
            todo.append((r, source_for_record(cfg, r, resolver=resolver)))
        except (OSError, ValueError, KeyError) as exc:
            unresolved += 1
            print(f"[review] {r['id']}: source unresolved ({exc}); left for verify to hold as pending")
    print(f"[review] {len(rows)} generated, {len(done)} already reviewed, {len(todo)} to review, {unresolved} unresolved")
    if reviewer is None:
        from .teacher import Teacher
        reviewer = Teacher(cfg.judge, name="reviewer")
    protocol = cfg.verify.grounding_review_protocol
    out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with ThreadPoolExecutor(cfg.judge.concurrency) as pool, out.open("a") as f:
        futures = [pool.submit(request_review, reviewer, r, src, protocol=protocol) for r, src in todo]
        for fut in as_completed(futures):
            try:
                report = fut.result()
            except Exception as exc:  # one failed request must not lose the others; rerun retries it
                failed += 1
                print(f"[review] request failed: {exc!r}")
                continue
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
            f.flush()
    print(f"[review] wrote {len(todo) - failed} reviews ({failed} failed, retried on rerun) -> {out}")
    return out
