"""Mean ± spread over training seeds (GOAL.md §7: every score over >= 3 seeds).

    python -m distillkit.aggregate runs/qwen3_4b_qdora [--tag gemma4]

Reads <run_dir>/seed_*/eval_summary[_<tag>].json, writes <run_dir>/eval_seeds[_<tag>].json and prints
a table. Numbers inside per-model sections ("base", "student") are kept apart: "student.tool_ast_acc".
"""

import argparse
import json
import statistics
from pathlib import Path


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= flatten(v, f"{key}.")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def aggregate(run_dir: Path, tag: str = "") -> dict:
    name = f"eval_summary{'_' + tag if tag else ''}.json"
    files = sorted(run_dir.glob(f"seed_*/{name}"))
    if not files:
        raise SystemExit(f"no {name} under {run_dir}/seed_*")
    # Include prepared seed directories with missing summaries: they are not successful seeds.
    summaries = {d.name: json.loads((d / name).read_text()) if (d / name).exists() else {}
                 for d in sorted(run_dir.glob("seed_*")) if d.is_dir()}
    per_seed = {seed: flatten(summary) for seed, summary in summaries.items()}
    protocols = {seed: summary.get("judge_protocol", "legacy-unversioned") for seed, summary in summaries.items()}
    judges = {seed: summary.get("judge", "unknown") for seed, summary in summaries.items()}
    win_keys = {k for summary in summaries.values() for k in summary if k.endswith("_win_rate")}
    keys = sorted(set().union(*per_seed.values()) | win_keys)
    stats = {}
    for k in keys:
        vals = [s[k] for s in per_seed.values() if k in s]
        if k in win_keys:
            reasons = []
            incomplete = [seed for seed, s in per_seed.items() if k not in s]
            if incomplete:
                reasons.append("missing or invalid seed score")
            if len(set(protocols.values())) != 1:
                reasons.append("mixed judge protocols")
            if len(set(judges.values())) != 1:
                reasons.append("mixed judge identities")
            if reasons:
                stats[k] = {"mean": None, "std": None, "min": None, "max": None,
                            "n": len(vals), "expected_n": len(per_seed), "status": "incomplete",
                            "reasons": reasons, "incomplete_seeds": incomplete}
                continue
        stats[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return {"seeds": list(per_seed), "metrics": stats, "judge_protocols": protocols, "judges": judges}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    res = aggregate(a.run_dir, a.tag)
    (a.run_dir / f"eval_seeds{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=2))
    print(f"seeds: {', '.join(res['seeds'])}")
    for k, s in res["metrics"].items():
        if s["mean"] is None:
            print(f"{k:45s} INCOMPLETE: {', '.join(s['reasons'])}  valid={s['n']}/{s['expected_n']}")
            continue
        print(f"{k:45s} {s['mean']:8.3f} ± {s['std']:.3f}   [{s['min']:.3f}, {s['max']:.3f}]  n={s['n']}")


if __name__ == "__main__":
    main()
