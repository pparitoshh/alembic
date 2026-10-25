"""Stage `review`: an independent model checks every generated record against its source.

Runs the existing support review (`verify.grounding_review_protocol`, e.g. source-support-v3) on
each row of run_dir/generated.jsonl, using the `judge` endpoint as the reviewer (serve a model other
than the teacher, e.g. --set judge.model=google/gemma-4-26B-A4B-it). Reports are appended to
`verify.grounding_reviews` (default run_dir/grounding_reviews.jsonl); rows already reviewed are
skipped and malformed reviews (e.g. truncated JSON) redone, so a rerun resumes and repairs.
`verify` with `require_grounding_review: true` then rejects unsupported answers and holds uncertain ones as pending. Rows whose source cannot be resolved are
not reviewed; verify holds them as pending.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import Config
from .io import read_jsonl, write_jsonl
from .support import request_review, source_for_record


REDO_TEMPERATURE = 0.7  # greedy decoding would repeat the same slip


def evidence_gap(report: dict) -> bool:
    """A unit judged supported with no cited evidence: a protocol violation that verify can only hold as pending."""
    units = (report.get("review") or {}).get("units") or []
    return any(u.get("verdict") == "supported" and not u.get("evidence_ids") for u in units)


def reviews_path(cfg: Config) -> Path:
    return cfg.verify.grounding_reviews or cfg.run_dir / "grounding_reviews.jsonl"


def run(cfg: Config, reviewer=None) -> Path:
    out = reviews_path(cfg)
    rows = read_jsonl(cfg.run_dir / "generated.jsonl")
    existing = read_jsonl(out) if out.exists() else []
    parsed = [r for r in existing if r.get("review") is not None]
    if len(parsed) < len(existing):  # malformed (e.g. truncated) reviews are redone, not kept as final
        write_jsonl(out, parsed)
        print(f"[review] dropped {len(existing) - len(parsed)} malformed reviews to redo them")
    # Evidence-gap reviews get exactly one recorded redo; the first attempt is archived, never discarded.
    archive = out.with_name(out.stem + "_attempt1.jsonl")
    # Opt-in: on the 2026-10-18 factory batch only 15% of redos came back with evidence, 80% repeated the gap.
    redo = os.environ.get("DISTILLKIT_REDO_EVIDENCE_GAPS") == "1"
    gaps = [r for r in parsed if redo and evidence_gap(r) and r.get("review_attempt", 1) < 2]
    if gaps:
        with archive.open("a") as f:
            for r in gaps:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dropped = {r["id"] for r in gaps}
        parsed = [r for r in parsed if r["id"] not in dropped]
        write_jsonl(out, parsed)
        print(f"[review] redoing {len(gaps)} reviews that marked a unit supported without evidence (archived to {archive.name})")
    # Archived ids still lacking their redo (e.g. the previous job timed out) stay on attempt 2.
    gap_ids = {r["id"] for r in read_jsonl(archive)} if redo and archive.exists() else set()
    done = {r["id"] for r in parsed}
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
        futures = [pool.submit(request_review, reviewer, r, src, protocol=protocol,
                               **({"temperature": REDO_TEMPERATURE, "attempt": 2} if r["id"] in gap_ids else {}))
                   for r, src in todo]
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
