"""Collect completed, correctly bound B6 evaluations; never publish partial win rates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from distillkit.eval_v3 import check_release, sha
from distillkit.io import read_jsonl


def checked_result(directory, expected_rows):
    summary = json.loads((directory / "eval_summary.json").read_text())
    outputs = read_jsonl(directory / "eval_outputs.jsonl")
    expected = {r["id"]: r for r in expected_rows}
    if len(outputs) != len(expected) or {r["id"] for r in outputs} != set(expected):
        raise ValueError("incomplete or different question IDs")
    for r in outputs:
        if any(r[k] != expected[r["id"]][k] for k in ("question", "reference", "doc_id", "task")):
            raise ValueError("evaluated question/reference differs from release")
        if not isinstance(r.get("answer_base"), str) or not isinstance(r.get("answer_student"), str):
            raise ValueError("missing model answer")
        verdicts = r.get("judge", {}).get("student_vs_base", [])
        if len(verdicts) != 2 or any(v not in {"A", "B", "T"} for v in verdicts):
            raise ValueError("incomplete paired verdicts")
    if (summary.get("judge_protocol") != "paired-orders-v2-invalid-incomplete"
            or summary.get("judge") != "openai/gpt-oss-20b"):
        raise ValueError("different judge protocol or model")
    comparison = summary["judge_comparisons"]["student_vs_base"]
    if (comparison["status"] != "complete" or comparison["questions"] != len(expected)
            or comparison["complete_pairs"] != len(expected) or comparison["invalid_verdicts"]):
        raise ValueError("incomplete summary")
    scores = {r["id"]: ({"A":1, "T":.5, "B":0}[r["judge"]["student_vs_base"][0]] +
                       {"B":1, "T":.5, "A":0}[r["judge"]["student_vs_base"][1]]) / 2 for r in outputs}
    rate = statistics.mean(scores.values())
    if abs(rate - summary["student_vs_base_win_rate"]) > 1e-12:
        raise ValueError("headline win rate does not match per-question verdicts")
    return summary, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/eval_v3_b6"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "incomplete", "v2": {}, "v3": {}, "errors": []}
    try:
        manifest = check_release()
        report["eval_v3_sha256"] = manifest["sha256"]
        v3_rows = read_jsonl(Path("data/eval/eval_v3.jsonl"))
        evidence = json.loads(Path(manifest["evidence_path"]).read_text())
        topics = {r["id"]: r["topic"] for r in evidence}
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(f"release: {exc}")
        v3_rows, topics = [], {}
    v2_rows = read_jsonl(Path("data/eval/eval_all_v2.jsonl"))
    for name, root, rows in [("v2", args.v2_root, v2_rows), ("v3", args.output, v3_rows)]:
        if not rows:
            continue
        for seed in (42, 1, 2):
            try:
                directory = root / f"seed_{seed}"
                summary, scores = checked_result(directory, rows)
                result = {"summary": summary, "outputs_sha256": sha((directory / "eval_outputs.jsonl").read_bytes())}
                if name == "v3":
                    grouped = defaultdict(list)
                    for rid, score in scores.items():
                        grouped[topics[rid]].append(score)
                    result["topics"] = {t: {"questions": len(values), "win_rate": statistics.mean(values)}
                                        for t, values in sorted(grouped.items())}
                report[name][str(seed)] = result
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report["errors"].append(f"{name} seed {seed}: {exc}")
    audit = Path("runs/eval_v3_audit/audit.json")
    if audit.exists():
        report["training_audit"] = json.loads(audit.read_text())
    else:
        report["errors"].append("training decontamination audit is not available")
    if not report["errors"] and len(report["v3"]) == len(report["v2"]) == 3:
        report["status"] = "complete"
    lines = ["# Eval v3 development comparison", "", f"Status: **{report['status']}**.", "",
        "V2 and v3 measure different topic distributions; compare models within each set.",
        "V3 uses source chunks already encountered in historical B6 training. Future training must exclude them.",
        "No model-quality improvement is implied by finishing this evaluation.", "",
        "| Set | Seed | Questions | Student vs base win rate | Base check pass | Student check pass |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ("v2", "v3"):
        for seed, result in report[name].items():
            s = result["summary"]
            lines.append(f"| {name} | {seed} | {s['judge_comparisons']['student_vs_base']['questions']} | "
                         f"{s['student_vs_base_win_rate']:.4f} | {s['base']['check_pass_rate']:.4f} | {s['student']['check_pass_rate']:.4f} |")
    if len(report["v3"]) == 3:
        lines += ["", "| V3 topic | Questions per seed | Mean win rate across 3 seeds |", "|---|---:|---:|"]
        first = next(iter(report["v3"].values()))["topics"]
        for topic in sorted(first):
            rates = [r["topics"][topic]["win_rate"] for r in report["v3"].values()]
            lines.append(f"| {topic} | {first[topic]['questions']} | {statistics.mean(rates):.4f} |")
    lines += ["", "These are descriptive scores. Shared sources, repeated seeds, and the two judge orders are correlated; "
              "do not treat their verdicts as independent trials or infer a statistically established improvement from a small difference."]
    if report["errors"]:
        lines += ["", "Outstanding:", "", *[f"- {error}" for error in report["errors"]]]
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
