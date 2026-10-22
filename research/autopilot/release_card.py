"""Bind staged release assets and documentation to the exact winning arm."""
import json
from pathlib import Path
import sys

from evidence import EvidenceError, objective_decision, read_json, sha256
from supervisor import atomic


def approved(config, run):
    code = Path(config["code"])
    saved = read_json(Path(config["root"]) / "ctl/autopilot/decisions" / (run + ".json"))
    current = objective_decision(code, run, config)
    checks = read_json(code / "runs" / run / "AUTOPILOT_CHECKS.json")
    if not current["goal_met"] or current != saved or not checks["passed"] or checks["arm"] != current["selected_arm"]:
        raise EvidenceError("Release approval evidence is missing, changed or failed")
    return current


def main(action, config_path, run):
    config = read_json(config_path)
    decision = approved(config, run)
    if action == "approved-arm":
        print(decision["selected_arm"])
        return
    if action != "write":
        raise ValueError(action)
    code = Path(config["code"])
    out = code / "runs" / f"hf_{run}"
    metrics = decision["candidates"][decision["selected_arm"]]["metrics"]
    lines = ["---", "base_model: Qwen/Qwen3-4B-Instruct-2507", "language: en", "pipeline_tag: text-generation", "---", "",
        "# Compact HPC assistant — evaluated candidate", "", f"Run: `{run}`; arm: `{decision['selected_arm']}`.", "",
        "This model starts from release 12e and uses rejection-sampled, source-grounded student answers from the frozen teacher bank. Three training seeds are evaluated; the selected arm is reported explicitly.", "",
        "## Development comparison with release 12e", "",
        "Differences below are paired judge preference scores against the same baseline, not factual-accuracy percentages.", "",
        "| Judge | v2 | v3 | d1 |", "|---|---:|---:|---:|"]
    for judge, slices in metrics.items():
        lines.append("| " + judge + " | " + " | ".join(f"{slices[s]['difference']:+.4f}" for s in ("v2", "v3", "d1")) + " |")
    lines.extend(["", "## Evaluation scope", "",
        "Both judges must show at least 90% preference on each development slice and first place by the unweighted three-slice mean against the frozen 13-model internal leaderboard. Invalid competitor judgments are conservatively scored as wins when establishing the target. Improvement over 12e with complete valid coverage, the existing v3 significance check, and tool/command non-regression checks must also pass. These development sets have been used repeatedly for model selection. The independent final holdout has not been run, and these results do not establish a world-best claim.", "",
        "The tool check covers 23 first-response simulator tasks, not full multi-turn work on a live cluster. Quantization comparisons and per-case failures accompany this candidate. Laptop RAM, latency and tokens per second still need measurement on the deployment device.", "",
        "## Artifacts", "", "The staged directory contains Hugging Face weights, Q8_0 and importance-matrix Q4_K_M GGUF files, the Ollama Modelfile, and both-judge comparison evidence. The automated service stages these files without uploading them or replacing the published model.", ""])
    (out / "README.md").write_text("\n".join(lines))
    atomic(out / "AUTOPILOT_DECISION.json", decision)
    files = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("READY", "MANIFEST.json")}
    atomic(out / "MANIFEST.json", {"run": run, "arm": decision["selected_arm"], "files": files})


if __name__ == "__main__":
    main(*sys.argv[1:])
