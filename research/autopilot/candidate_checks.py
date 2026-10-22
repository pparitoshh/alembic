"""Run the existing command and first-response tool checks before staging a benchmark winner."""
import json
import os
from pathlib import Path
import subprocess
import sys

from evidence import EvidenceError, read_json, rows
from supervisor import atomic


def main(config_path, run):
    config = read_json(config_path)
    code = Path(config["code"])
    runtime = Path(__file__).resolve().parent
    decision = read_json(Path(config["root"]) / "ctl/autopilot/decisions" / (run + ".json"))
    if not decision["goal_met"]:
        raise EvidenceError("Candidate has not passed the benchmark gate")
    arm = decision["selected_arm"]
    env = dict(os.environ, REPO=str(code), RUN=run, BASE_MODEL=str(code / config["baseline_model"]),
               PYTHONPATH=str(code / "src") + ":" + str(code / "research/rag65"))
    tool_file = code / "runs" / run / "TOOL_EVAL.json"
    if not tool_file.exists():
        subprocess.run([config["python"], str(runtime / "gpu_submit.py"), config_path,
                        "--export=ALL", str(runtime / "tool_eval.sbatch")], env=env, cwd=code, check=True)
    tools = read_json(tool_file)
    reference = tools[env["BASE_MODEL"]]
    candidate = tools[arm.removesuffix("_rag")]
    ids = set(rows(code / "data/eval/dataset_v1/development_tools.jsonl"))
    for entry in (reference, candidate):
        actual = [r["id"] for r in entry["results"]]
        if len(actual) != len(ids) or set(actual) != ids:
            raise EvidenceError("Tool evaluation coverage is incomplete")
    static_file = code / "runs" / run / "AUTOPILOT_STATIC.json"
    comparison = code / "runs" / f"vs_release_{run}"
    subprocess.run([config["python"], str(code / "research/rag65/static_checks.py"),
        str(comparison / "answers"), str(static_file), arm, "rel12e_rag"], env=env, cwd=code, check=True)
    static = read_json(static_file)
    reasons = []
    for metric in ("first_response_pass", "decision_ok", "valid_call_rate"):
        a, b = candidate.get(metric), reference.get(metric)
        if b is not None and (a is None or a < b):
            reasons.append(f"Tool {metric} regressed: {a} vs {b}")
    for label in ("v2", "v3", "d1"):
        a, b = static[f"{label}/{arm}"], static[f"{label}/rel12e_rag"]
        if a["check_pass"] < b["check_pass"] or a["bad_flags_per_answer"] > b["bad_flags_per_answer"]:
            reasons.append(f"{label}: command validity regressed")
    atomic(code / "runs" / run / "AUTOPILOT_CHECKS.json", {
        "passed": not reasons, "run": run, "arm": arm, "reasons": reasons,
        "tool_items": len(ids), "tool_scope": "First response against simulator; not full live-cluster task completion.",
        "tool_candidate": {k: v for k, v in candidate.items() if k != "results"},
        "tool_reference": {k: v for k, v in reference.items() if k != "results"}, "static": static})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
