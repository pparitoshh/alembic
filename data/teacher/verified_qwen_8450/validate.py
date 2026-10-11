"""Validate the immutable 8,450-row teacher snapshot."""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "verified.jsonl"
REPORT = ROOT / "certification_report.json"
EXPECTED_ROWS = 8450
EXPECTED_BASE = 4775
EXPECTED_NEW = 3675
BASE = ROOT.parent / "verified_qwen_4775" / "verified.jsonl"
NEW_PREFIX = "newdocs-second-20261010/"


def normalized_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "AUTOMATED_GATE_CERTIFIED":
        raise ValueError("The corrected certificate is not in the accepted state")
    if report.get("combined") != EXPECTED_ROWS or report.get("new_verified_under_automated_gate") != EXPECTED_NEW:
        raise ValueError("Certificate counts differ from this release")
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != report.get("combined_sha256"):
        raise ValueError(f"Dataset SHA-256 mismatch: {digest}")
    if report.get("base_accepted") != EXPECTED_BASE or not DATA.read_bytes().startswith(BASE.read_bytes()):
        raise ValueError("The prior certified release is not an exact prefix")

    ids: set[str] = set()
    questions: set[str] = set()
    answers: set[str] = set()
    new_count = 0
    with DATA.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            row = json.loads(line)
            row_id = row.get("id")
            question = row.get("question")
            messages = row.get("messages")
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise ValueError(f"Missing or repeated ID on line {number}")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question on line {number}")
            key = normalized_question(question)
            if key in questions:
                raise ValueError(f"Repeated normalized question on line {number}")
            if not isinstance(messages, list) or not any(
                isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str) and message["content"].strip()
                for message in messages
            ):
                raise ValueError(f"Missing assistant answer on line {number}")
            answer = next(message["content"] for message in reversed(messages) if message.get("role") == "assistant")
            answer_key = normalized_question(answer)
            if answer_key in answers:
                raise ValueError(f"Repeated normalized answer on line {number}")
            ids.add(row_id)
            questions.add(key)
            answers.add(answer_key)
            if row_id.startswith(NEW_PREFIX):
                new_count += 1
                if not str(row.get("source_origin", "")).startswith("https://github.com/"):
                    raise ValueError(f"New row lacks pinned source origin on line {number}")
    if len(ids) != EXPECTED_ROWS or len(answers) != EXPECTED_ROWS or new_count != EXPECTED_NEW:
        raise ValueError(f"Unexpected release composition: {len(ids)} rows, {new_count} new")
    print(f"Validated {len(ids)} distinct Q&A; {new_count} new; SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
