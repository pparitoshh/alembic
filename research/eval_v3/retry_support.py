"""Bounded retries of structurally invalid positive support reviews only.

The support protocol, prompt, evidence and validator are unchanged. Constrain
transport IDs to IDs that actually exist in the packet. Preserve every original
and retry report; never reroll negative or genuinely uncertain judgments.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json

from distillkit.config import load_config
from distillkit.eval_v3 import preflight, unique_rows
from distillkit.io import read_jsonl, write_jsonl
from distillkit.support import digest, source_for_record
from distillkit import support_v3
from distillkit.teacher import Teacher

STRUCTURAL_ERRORS = {
    "malformed review", "citation absent, altered or unavailable before this unit",
    "unknown or duplicate target unit", "supported unit lacks evidence",
    "unreviewed assistant target units",
}


def retryable(report, result):
    if result.get("status") != "uncertain" or result.get("reason") not in STRUCTURAL_ERRORS:
        return False
    review = report.get("review") or {}
    if review.get("uncertainty", {}).get("present"):
        return False
    return not any(u.get("verdict") in {"unsupported", "uncertain"} for u in review.get("units", []))


def constrained_schema(schema, packet):
    schema = copy.deepcopy(schema)
    item = schema["$defs"]["UnitReview"]["properties"]
    item["unit_id"]["enum"] = [u["unit_id"] for u in packet["units"]]
    item["evidence_ids"]["items"]["enum"] = sorted({
        eid for turn in packet["turns"] for eid in turn["available_evidence_ids"]})
    schema["properties"]["units"].update(minItems=len(packet["units"]), maxItems=len(packet["units"]))
    return schema


class BoundIDReviewer:
    def __init__(self, client):
        self.client, self.model, self.schema_sha256 = client, client.model, None

    def complete(self, messages, **kwargs):
        fmt = copy.deepcopy(kwargs["response_format"])
        schema = constrained_schema(fmt["json_schema"]["schema"], json.loads(messages[-1]["content"]))
        fmt["json_schema"]["schema"] = schema
        self.schema_sha256 = digest(schema)
        return self.client.complete(messages, **{**kwargs, "response_format": fmt})


def atomic_write(path, rows):
    temporary = path.with_suffix(".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    cfg = load_config("configs/gen_eval_v3.yaml", [f"run_dir={args.run_dir}"])
    preflight(cfg)
    endpoint = cfg.judge.model_copy(update={"base_url": args.url, "model": "google/gemma-4-26B-A4B-it",
        "max_tokens": 8192, "temperature": 0, "extra_body": {"repetition_penalty": 1.05}})
    client = Teacher(endpoint, name="support-format-retry")
    reports_path = cfg.run_dir / "grounding_reviews.jsonl"
    history_path = cfg.run_dir / "support_retry_history.jsonl"
    reports = unique_rows(reports_path)
    rows = unique_rows(cfg.run_dir / "generated.jsonl")
    pending = unique_rows(cfg.run_dir / "pending_review.jsonl")
    history = read_jsonl(history_path) if history_path.exists() else []
    attempts = Counter(h["id"] for h in history)
    # Recover a persisted response if a job stopped between history and state writes.
    for item in history:
        if digest(reports[item["id"]]) == item["before_sha256"]:
            reports[item["id"]] = item["after"]
    todo, skipped = [], Counter()
    for rid in pending:
        row = rows[rid]
        source = source_for_record(cfg, row)
        report = reports[rid]
        result = support_v3.check_support(row, source, report)
        if attempts[rid] >= 2:
            skipped["retry_limit"] += 1
        elif retryable(report, result):
            todo.append((row, source, report, result["reason"]))
        else:
            skipped["not_a_retryable_format_error"] += 1
    print(f"[support retry] {len(todo)} eligible, skipped {dict(skipped)}", flush=True)

    def request(item):
        row, source, before, reason = item
        reviewer = BoundIDReviewer(client)
        after = support_v3.request_review(reviewer, row, source)
        after["transport_schema_sha256"] = reviewer.schema_sha256
        return {"id": row["id"], "attempt": attempts[row["id"]] + 1, "reason": reason,
            "before_sha256": digest(before), "before": before, "after": after,
            "after_check": support_v3.check_support(row, source, after)}

    with ThreadPoolExecutor(cfg.judge.concurrency) as pool:
        for future in as_completed([pool.submit(request, item) for item in todo]):
            item = future.result()
            history.append(item)
            atomic_write(history_path, history)
            reports[item["id"]] = item["after"]
            atomic_write(reports_path, [reports[k] for k in sorted(reports)])
    atomic_write(reports_path, [reports[k] for k in sorted(reports)])
    print(f"[support retry] completed {len(todo)} requests", flush=True)


if __name__ == "__main__":
    main()
