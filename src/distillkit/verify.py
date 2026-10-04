"""Stage 2: dedup -> deterministic checks -> decontamination against the eval set."""

import re
from collections import Counter
from pathlib import Path

from .checks import check_answer, load_flags
from .config import Config
from .io import read_jsonl, write_jsonl
from .records import final_answer


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def run(cfg: Config) -> Path:
    vcfg, run_dir = cfg.verify, cfg.run_dir
    # generate appends rows in completion order; sort so "first copy wins" in dedup is reproducible
    rows = sorted(read_jsonl(run_dir / "generated.jsonl"), key=lambda r: r.get("id", ""))
    valid_flags = load_flags(vcfg.flag_list)
    eval_shingles = [_shingles(r["question"]) for r in read_jsonl(cfg.eval.file)]

    kept, rejected = [], []
    kept_questions: dict[str, set] = {}  # question text -> shingles (N answers share one question)

    for r in rows:
        reason = None
        q, a = r["question"], final_answer(r)
        if not q or not a or len(a) > vcfg.max_answer_chars:
            reason = "empty_or_too_long"
        if reason is None and q not in kept_questions:
            sh = _shingles(q)
            if any(_jaccard(sh, e) >= vcfg.dedup_threshold for e in eval_shingles):
                reason = "eval_contamination"
            elif any(_jaccard(sh, k) >= vcfg.dedup_threshold for k in kept_questions.values()):
                reason = "near_duplicate"
            else:
                kept_questions[q] = sh
        if reason is None:
            checks = check_answer(a, valid_flags)
            r = {**r, "checks": checks}
            if checks["bad_flags"]:
                reason = "bad_flags"
            elif not checks["bash_ok"]:
                reason = "bash_syntax"
        if reason:
            rejected.append({**r, "reject_reason": reason})
        else:
            kept.append(r)

    write_jsonl(run_dir / "verified.jsonl", kept)
    write_jsonl(run_dir / "rejected.jsonl", rejected)
    reasons = Counter(r["reject_reason"] for r in rejected)
    print(f"[verify] kept {len(kept)}/{len(rows)}; rejected: {dict(reasons)}")
    return run_dir / "verified.jsonl"
