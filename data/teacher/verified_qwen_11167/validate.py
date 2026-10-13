"""Validate the immutable 11,167-row corrected teacher snapshot."""

import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "verified.jsonl"
BASE = ROOT.parent / "verified_qwen_8450" / "verified.jsonl"
COHORTS = (
    ("third", 9736, 1286, "newdocs-third-20261010/"),
    ("fourth", 9985, 249, "newdocs-fourth-20261010/"),
    ("fifth", 11167, 1182, "newdocs-fifth-20261010/"),
)
PINNED_URL = re.compile(r"https://github\.com/[^/]+/[^/]+/blob/[0-9a-f]{40}/.+")
SHA256 = re.compile(r"[0-9a-f]{64}")


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def main() -> int:
    data = DATA.read_bytes()
    base = BASE.read_bytes()
    if not data.startswith(base):
        raise ValueError("The prior 8,450-row certified release is not an exact prefix")
    lines = data.splitlines(keepends=True)
    if len(lines) != 11167:
        raise ValueError(f"Expected 11,167 rows, found {len(lines)}")

    prior_count = 8450
    for name, count, added, prefix in COHORTS:
        report_path = ROOT / ("certification_report.json" if name == "fifth" else f"certification_report_{name}.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("status") != "AUTOMATED_GATE_CERTIFIED"
                or report.get("base_accepted") != prior_count
                or report.get("new_verified_under_automated_gate") != added
                or report.get("combined") != count):
            raise ValueError(f"{name} corrected certificate counts/status differ")
        digest = hashlib.sha256(b"".join(lines[:count])).hexdigest()
        if digest != report.get("combined_sha256"):
            raise ValueError(f"{name} corrected certificate hash mismatch: {digest}")
        for number, line in enumerate(lines[prior_count:count], start=prior_count + 1):
            row = json.loads(line)
            if not row.get("id", "").startswith(prefix):
                raise ValueError(f"Unexpected {name} row ID on line {number}")
            if not PINNED_URL.fullmatch(str(row.get("source_origin", ""))):
                raise ValueError(f"Missing pinned GitHub source on line {number}")
            for key in ("source_sha256", "document_sha256", "independent_review_result_sha256"):
                if not SHA256.fullmatch(str(row.get(key, ""))):
                    raise ValueError(f"Missing {key} on line {number}")
        prior_count = count

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
        ids.add(row_id)
        questions.add(qkey)
        answers.add(akey)

    print(f"Validated {len(ids):,} distinct Q&A; 2,717 new; SHA-256 {hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
