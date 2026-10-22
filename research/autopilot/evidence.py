"""Read-only evidence checks. A log line is never a completion or promotion signal."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SLICES = {
    "v2": "data/eval/eval_all_v2.jsonl",
    "v3": "data/eval/eval_v3.jsonl",
    "d1": "data/eval/dataset_v1/development_normal.jsonl",
}
SEEDS = (42, 43, 44)
ARMS = [f"s{s}{variant}_rag" for s in SEEDS for variant in ("", "_x05")]


class EvidenceError(ValueError):
    pass


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"Cannot read {path}: {exc}") from exc


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    result = {}
    try:
        with Path(path).open() as f:
            for number, line in enumerate(f, 1):
                row = json.loads(line)
                rid = row["id"]
                if not isinstance(rid, str) or rid in result:
                    raise EvidenceError(f"Duplicate/invalid ID at {path}:{number}")
                result[rid] = row
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvidenceError(f"Invalid JSONL {path}: {exc}") from exc
    return result


def expected_ids(code):
    return {label: set(rows(Path(code) / path)) for label, path in SLICES.items()}


def answer_rows(path, expected, *, allow_empty=False):
    data = rows(path)
    if set(data) != expected:
        raise EvidenceError(f"{path}: expected {len(expected)} exact question IDs, got {len(data)}")
    for rid, row in data.items():
        if not isinstance(row.get("answer"), str) or (not allow_empty and not row["answer"].strip()):
            raise EvidenceError(f"{path}: empty answer {rid}")
        if row.get("finish_reason") not in ("stop", "length"):
            raise EvidenceError(f"{path}: unexpected finish reason for {rid}")
    return data


def audit_round(code, run):
    root = Path(code) / "runs" / run
    expected = expected_ids(code)
    report = read_json(root / "RESULT.json")
    count, truncated, invalid = 0, 0, 0
    for seed in SEEDS:
        read_json(root / f"train/seed{seed}/TRAINING_DONE.json")
        if not (root / f"train/seed{seed}/adapter/adapter_model.safetensors").is_file():
            raise EvidenceError(f"{run}: seed {seed} adapter is missing")
        for variant in ("", "_x05", "_x02"):
            for mode in ("closed", "rag"):
                arm = f"s{seed}{variant}_{mode}"
                for label, ids in expected.items():
                    data = answer_rows(root / "answers" / f"{label}__{arm}.jsonl", ids)
                    metric = report.get("arms", {}).get(f"{label}/{arm}")
                    if not isinstance(metric, dict):
                        raise EvidenceError(f"{run}: missing aggregated metric {label}/{arm}")
                    count += len(data)
                    truncated += sum(r["finish_reason"] == "length" for r in data.values())
                    invalid += metric.get("invalid", 0)
    # Invalid verdicts/truncated answers remain visible. They do not become missing work to retry away.
    return {"answers": count, "truncated": truncated, "invalid_verdicts": invalid,
            "result_sha256": sha256(root / "RESULT.json")}


def audit_factory(code, batch):
    root = Path(code) / "runs" / batch / "release"
    report = read_json(root / "REPORT.json")
    data = root / "verified.jsonl"
    if report.get("status") != "automated_gates_passed_not_human_reviewed":
        raise EvidenceError(f"{batch}: missing automated-gate status")
    if sha256(data) != report.get("sha256"):
        raise EvidenceError(f"{batch}: release digest does not match REPORT.json")
    with data.open() as f:
        count = sum(1 for line in f if line.strip())
    if count != report.get("released_rows") or count <= 0:
        raise EvidenceError(f"{batch}: release count mismatch")
    return {"released_rows": count, "prior_bank_rows": report.get("bank_rows"),
            "sha256": report["sha256"], "human_reviewed": False}


def audit_judgments(code, directory, judge_dir, arms, *, allow_empty=False):
    """Check exact coverage and bind every judgment to the saved candidate AND baseline bytes."""
    directory = Path(directory)
    expected = expected_ids(code)
    result = {}
    for label, ids in expected.items():
        base = answer_rows(directory / "answers" / f"{label}__base_closed.jsonl", ids)
        for arm in arms:
            answers = answer_rows(directory / "answers" / f"{label}__{arm}.jsonl", ids, allow_empty=allow_empty)
            seen, scores, invalid = set(), {}, 0
            for path in sorted((directory / judge_dir).glob(f"{label}__{arm}__*.jsonl")):
                for rid, row in rows(path).items():
                    if rid not in ids or rid in seen:
                        raise EvidenceError(f"{path}: unexpected/duplicate judgment {rid}")
                    seen.add(rid)
                    if row.get("answer_arm") != answers[rid]["answer"] or row.get("answer_base") != base[rid]["answer"]:
                        raise EvidenceError(f"{path}: judgment answer binding mismatch for {rid}")
                    verdicts = row.get("judge", {}).get("arm_vs_base")
                    if not isinstance(verdicts, list) or len(verdicts) != 2 or any(v not in ("A", "B", "T", None) for v in verdicts):
                        raise EvidenceError(f"{path}: malformed verdicts for {rid}")
                    if None in verdicts:
                        invalid += 1
                    else:
                        scores[rid] = ({"A": 1, "B": 0, "T": .5}[verdicts[0]] +
                                       {"B": 1, "A": 0, "T": .5}[verdicts[1]]) / 2
            if seen != ids:
                raise EvidenceError(f"{directory.name}/{judge_dir}/{label}/{arm}: {len(seen)}/{len(ids)} judgments")
            result[f"{label}/{arm}"] = {"scores": scores, "invalid": invalid,
                "truncated": sum(r["finish_reason"] == "length" for r in answers.values())}
    return result


def promotion_decision(code, run, reference="rel12e_rag", alpha=.05):
    """Development-only stop rule: all three slices positive under both judges; retain the v3 significance gate."""
    directory = Path(code) / "runs" / f"vs_release_{run}"
    expected = expected_ids(code)
    paired = {"gpt_oss": read_json(directory / "PAIRED.json"),
              "gemma": read_json(directory / "PAIRED_gemma.json")}
    raw = {"gpt_oss": audit_judgments(code, directory, "judged", [*ARMS, reference]),
           "gemma": audit_judgments(code, directory, "judged_gemma", [*ARMS, reference])}
    candidates = {}
    for arm in ARMS:
        reasons, metrics, v3_ps = [], {}, []
        for judge, pairs in paired.items():
            metrics[judge] = {}
            for label, ids in expected.items():
                key = f"{label}/{arm}-vs-{reference}"
                pair = pairs.get(key, {})
                a, b = raw[judge][f"{label}/{arm}"], raw[judge][f"{label}/{reference}"]
                valid = set(a["scores"]) & set(b["scores"])
                if valid != ids or pair.get("questions") != len(ids):
                    reasons.append(f"{judge}/{label}: incomplete or invalid judgments")
                diff, p = pair.get("diff"), pair.get("p_a_not_better")
                if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (diff, p)) or not 0 <= p <= 1:
                    reasons.append(f"{judge}/{label}: invalid paired statistics")
                    continue
                actual = sum(a["scores"][i] - b["scores"][i] for i in valid) / len(valid) if valid else None
                if actual is None or abs(actual - diff) > .00011:
                    raise EvidenceError(f"{key}: paired statistic disagrees with raw judgments")
                if diff <= 0:
                    reasons.append(f"{judge}/{label}: no improvement ({diff:+.4f})")
                if label == "v3":
                    v3_ps.append(p)
                metrics[judge][label] = {"difference": diff, "p_a_not_better": p,
                    "score": sum(a["scores"].values()) / len(a["scores"]) if a["scores"] else None,
                    "questions": len(valid), "invalid": a["invalid"], "truncated": a["truncated"]}
        if len(v3_ps) != 2 or min(v3_ps) >= alpha:
            reasons.append(f"v3: existing significance threshold {alpha} not met")
        candidates[arm] = {"passed": not reasons, "reasons": reasons, "metrics": metrics}
    passing = [a for a in ARMS if candidates[a]["passed"]]
    selected = max(passing, key=lambda a: min(sum(m["difference"] for m in j.values()) / 3
                   for j in candidates[a]["metrics"].values())) if passing else None
    return {"run": run, "reference": reference, "goal_met": bool(selected), "selected_arm": selected,
        "scope": "Repeated development evaluation; final holdout remains untouched; not an accuracy estimate.",
        "rule": "Positive paired differences on v2, v3 and d1 under BOTH judges, full valid coverage, v3 p < alpha under at least one judge.",
        "alpha": alpha, "candidates": candidates,
        "evidence_sha256": {n: sha256(directory / n) for n in ("PAIRED.json", "PAIRED_gemma.json")}}


def answer_digest(data):
    return hashlib.sha256(json.dumps({k: v["answer"] for k, v in data.items()},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def leaderboard_snapshot(code):
    """Freeze raw, answer-bound standings. Empty competitor responses remain scored failures."""
    directory = Path(code) / "runs/leaderboard"
    ids = expected_ids(code)
    arms = sorted(p.name[len("v3__"):-6] for p in (directory / "answers").glob("v3__*.jsonl")
                  if p.name != "v3__base_closed.jsonl")
    if len(arms) < 13:
        raise EvidenceError("The current 13-model leaderboard is incomplete")
    result = {"arms": arms, "judges": {}, "baseline_answer_digests": {
        label: answer_digest(answer_rows(directory / "answers" / f"{label}__base_closed.jsonl", expected))
        for label, expected in ids.items()}}
    for judge, folder in (("gpt_oss", "judged"), ("gemma", "judged_gemma")):
        raw = audit_judgments(code, directory, folder, arms, allow_empty=True)
        table = {}
        for arm in arms:
            slices = {}
            for label, expected in ids.items():
                item = raw[f"{label}/{arm}"]
                total = sum(item["scores"].values())
                n = len(item["scores"])
                slices[label] = {"score": total / n if n else None, "valid": n,
                    "invalid": item["invalid"], "truncated": item["truncated"],
                    # Invalid competitor verdicts cannot make the target easier to beat.
                    "upper_score": (total + item["invalid"]) / len(expected)}
            table[arm] = {"slices": slices,
                "mean_upper": sum(s["upper_score"] for s in slices.values()) / len(SLICES)}
        result["judges"][judge] = table
    result["source_sha256"] = {str(p.relative_to(directory)): sha256(p)
        for folder in ("answers", "judged", "judged_gemma")
        for p in sorted((directory / folder).glob("*.jsonl"))}
    return result


def objective_decision(code, run, config):
    """The current goal supersedes the release checkpoint: 90+ everywhere and first place."""
    decision = promotion_decision(code, run, alpha=config["alpha"])
    target = config.get("target")
    if not target:
        return decision
    snapshot = config["leaderboard_snapshot"]
    directory = Path(code) / "runs" / f"vs_release_{run}"
    for label, ids in expected_ids(code).items():
        baseline = answer_rows(directory / "answers" / f"{label}__base_closed.jsonl", ids)
        if answer_digest(baseline) != snapshot["baseline_answer_digests"][label]:
            raise EvidenceError("Candidate comparison and leaderboard do not share the same baseline answers")
    decision["release_checkpoint_met"] = decision["goal_met"]
    decision["target"] = target
    decision["leaderboard"] = {}
    passing = []
    for arm, candidate in decision["candidates"].items():
        reasons, judges = [], {}
        for judge, metrics in candidate["metrics"].items():
            if set(metrics) != set(SLICES) or any(m["score"] is None for m in metrics.values()):
                reasons.append(f"{judge}: missing benchmark scores")
                continue
            scores = {label: m["score"] for label, m in metrics.items()}
            mean = sum(scores.values()) / len(scores)
            table = snapshot["judges"][judge]
            rank = 1 + sum(v["mean_upper"] >= mean for v in table.values())
            for label, score in scores.items():
                if score < target["minimum_score"]:
                    reasons.append(f"{judge}/{label}: {score:.4f} below {target['minimum_score']:.2f}")
            if rank != 1:
                reasons.append(f"{judge}: rank {rank}, first place required")
            judges[judge] = {"scores": scores, "mean": mean, "rank_conservative": rank}
        passed = candidate["passed"] and len(judges) == 2 and not reasons
        decision["leaderboard"][arm] = {"passed": passed, "judges": judges, "reasons": reasons}
        if passed:
            passing.append(arm)
    decision["selected_arm"] = max(passing, key=lambda arm: min(
        j["mean"] for j in decision["leaderboard"][arm]["judges"].values())) if passing else None
    decision["goal_met"] = bool(passing)
    decision["rule"] += " Additionally, every slice must score at least 90% and the candidate must rank first under both judges against the frozen 13-model leaderboard."
    decision["scope"] += " Leaderboard rank is internal and fixed to this protocol; it is not a global ranking."
    return decision
