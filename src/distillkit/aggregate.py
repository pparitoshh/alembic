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
    per_seed = {f.parent.name: flatten(json.loads(f.read_text())) for f in files}
    keys = sorted(set().union(*per_seed.values()))
    stats = {}
    for k in keys:
        vals = [s[k] for s in per_seed.values() if k in s]
        stats[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return {"seeds": list(per_seed), "metrics": stats}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    res = aggregate(a.run_dir, a.tag)
    (a.run_dir / f"eval_seeds{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=2))
    print(f"seeds: {', '.join(res['seeds'])}")
    for k, s in res["metrics"].items():
        print(f"{k:45s} {s['mean']:8.3f} ± {s['std']:.3f}   [{s['min']:.3f}, {s['max']:.3f}]  n={s['n']}")


if __name__ == "__main__":
    main()
