"""Validate the immutable 14,304-row corrected teacher snapshot."""

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "verified.jsonl"
BASE = ROOT.parent / "verified_qwen_12092" / "verified.jsonl"
REPORT = ROOT / "certification_report.json"
EXPECTED_ROWS = 14304
EXPECTED_NEW = 2212
EXPECTED_HASH = "c31c89a155d8b2cab46dd1f63387a120e2c542dec5c337622cdcba79c86af61d"
PINNED_URL = re.compile(r"https://github\.com/[^/]+/[^/]+/blob/[0-9a-f]{40}/.+")
SHA256 = re.compile(r"[0-9a-f]{64}")


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main() -> int:
    data, base = DATA.read_bytes(), BASE.read_bytes()
    if not data.startswith(base):
        raise ValueError("The prior 12,092-row release is not an exact prefix")
    lines = data.splitlines(keepends=True)
    if len(lines) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(lines)}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError(f"Release hash mismatch: {digest}")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if (report.get("status") != "AUTOMATED_GATE_CERTIFIED"
            or report.get("base_accepted") != 12092
            or report.get("new_verified_under_automated_gate") != EXPECTED_NEW
            or report.get("combined") != EXPECTED_ROWS
            or report.get("combined_sha256") != digest
            or report.get("teacher") != "Qwen/Qwen3-32B-AWQ"):
        raise ValueError("Corrected certificate does not bind this release")

    ids, questions, answers = set(), set(), set()
    for number, line in enumerate(lines, start=1):
        row = json.loads(line)
        row_id, question = row.get("id"), row.get("question")
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"Missing or repeated ID on line {number}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question on line {number}")
        qkey = normalized(question)
        if qkey in questions:
            raise ValueError(f"Repeated normalized question on line {number}")
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Missing messages on line {number}")
        assistant = [m.get("content") for m in messages
                     if isinstance(m, dict) and m.get("role") == "assistant"
                     and isinstance(m.get("content"), str) and m["content"].strip()]
        if not assistant:
            raise ValueError(f"Missing assistant answer on line {number}")
        akey = normalized(assistant[-1])
        if akey in answers:
            raise ValueError(f"Repeated normalized answer on line {number}")
        if row.get("teacher") != "Qwen/Qwen3-32B-AWQ":
            raise ValueError(f"Unexpected teacher on line {number}")
        if number > 12092:
            if not row_id.startswith("newdocs-eighth-20261010/"):
                raise ValueError(f"Unexpected eighth-cohort ID on line {number}")
            if not PINNED_URL.fullmatch(str(row.get("source_origin", ""))):
                raise ValueError(f"Missing pinned source URL on line {number}")
            for key in ("source_sha256", "document_sha256", "independent_review_result_sha256"):
                if not SHA256.fullmatch(str(row.get(key, ""))):
                    raise ValueError(f"Missing {key} on line {number}")
        ids.add(row_id)
        questions.add(qkey)
        answers.add(akey)

    print(f"Validated {len(ids):,} distinct Q&A; {EXPECTED_NEW:,} new; SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
