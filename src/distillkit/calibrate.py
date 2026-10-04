"""Stage `calibrate`: how far can we trust a judge? Run it on answer pairs whose verdict is known.

Each row of `eval.calibration_file` has a question, a reference, two answers and the correct label
(A, B or T). Every pair is judged in both orders, like the real eval, and we report:
- accuracy: share of judgements (both orders) that match the label;
- position consistency: share of pairs where swapping the answers swaps the verdict;
- accuracy per defect type (invented flag, wrong semantics, length bait, tie, ...).

Pick the judge with the best accuracy and consistency, and use it for every comparison.
"""

import json
from collections import defaultdict

from .config import Config
from .evaluate import judge
from .io import read_jsonl, write_jsonl
from .teacher import Teacher

SWAP = {"A": "B", "B": "A", "T": "T"}


def score(rows: list[dict], verdicts: list[tuple[str | None, str | None]]) -> dict:
    """verdicts[i] = (verdict with answers in file order, verdict with answers swapped)."""
    hits, consistent, unparsed = 0, 0, 0
    per_defect: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, (v1, v2) in zip(rows, verdicts):
        unparsed += (v1 is None) + (v2 is None)
        ok = (v1 == row["label"]) + (v2 == SWAP[row["label"]])  # 0, 1 or 2 correct judgements
        hits += ok
        consistent += v1 is not None and v2 == SWAP[v1]
        d = per_defect[row.get("defect", "other")]
        d[0] += ok
        d[1] += 2
    n = len(rows)
    return {
        "pairs": n,
        "accuracy": hits / (2 * n),
        "position_consistency": consistent / n,
        "unparsed": unparsed,
        "accuracy_by_defect": {k: round(h / t, 3) for k, (h, t) in sorted(per_defect.items())},
    }


def run(cfg: Config) -> dict:
    if not cfg.eval.calibration_file:
        raise SystemExit("[calibrate] set eval.calibration_file")
    rows = read_jsonl(cfg.eval.calibration_file)
    t = Teacher(cfg.judge, name="judge")
    jobs = [(r, a, b) for r in rows for a, b in ((r["answer_a"], r["answer_b"]), (r["answer_b"], r["answer_a"]))]
    out = t.map(lambda j: judge(t, *j), jobs)
    verdicts = [(out[2 * i][0], out[2 * i + 1][0]) for i in range(len(rows))]

    summary = {"judge": cfg.judge.model, **score(rows, verdicts)}
    suffix = f"_{cfg.eval.tag}" if cfg.eval.tag else ""
    write_jsonl(
        cfg.run_dir / f"calibration_outputs{suffix}.jsonl",
        [{**r, "verdicts": list(v), "raw": [out[2 * i][1], out[2 * i + 1][1]]} for i, (r, v) in enumerate(zip(rows, verdicts))],
    )
    (cfg.run_dir / f"calibration_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
