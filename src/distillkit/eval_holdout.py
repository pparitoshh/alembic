"""Explicit source exclusions in addition to lexical question decontamination.

A chunk split is a development split, not proof that historical models have
never seen its contents. IDs and hashes cover known source aliases; this is not
a semantic paraphrase detector.
"""
import hashlib
import json
from pathlib import Path
import re


def load_source_holdouts(root: Path) -> list[dict]:
    entries = []
    for path in sorted(root.rglob("*.holdout.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("version") != "eval-source-holdout-v1":
            raise ValueError(f"unknown source holdout version: {path}")
        for item in manifest["sources"]:
            if not isinstance(item.get("doc_id"), str) or not item["doc_id"]:
                raise ValueError(f"missing holdout document ID: {path}")
            if not re.fullmatch(r"[0-9a-f]{64}", item.get("source_sha256", "")):
                raise ValueError(f"invalid holdout source hash: {path}")
            if item.get("kind") not in {"chunk", "document"}:
                raise ValueError(f"invalid holdout source kind: {path}")
            if item["kind"] == "chunk" and not item.get("chunk_id"):
                raise ValueError(f"missing holdout chunk ID: {path}")
            if "text" in item and hashlib.sha256(item["text"].encode()).hexdigest() != item["source_sha256"]:
                raise ValueError(f"holdout text/hash mismatch: {path}")
            entries.append({**item, "manifest": str(path.relative_to(root))})
    return entries


def source_overlap(row: dict, held: list[dict]) -> dict | None:
    for item in held:
        same_doc = row.get("doc_id") == item["doc_id"]
        same_hash = row.get("source_sha256") == item["source_sha256"]
        same_chunk = same_doc and row.get("chunk_id") == item.get("chunk_id")
        # Full-document or unbound records may have consumed the held chunk even
        # when the stored question discusses another paragraph. Drop conservatively.
        unbound = same_doc and (row.get("source_kind") != "chunk" or
                               not row.get("chunk_id") or
                               not re.fullmatch(r"[0-9a-f]{64}", row.get("source_sha256", "")))
        if same_hash or (same_doc and item["kind"] == "document") or same_chunk or unbound:
            return {"manifest": item["manifest"], "doc_id": item["doc_id"],
                    "chunk_id": item.get("chunk_id"), "source_sha256": item["source_sha256"],
                    "reason": "heldout_source"}
    return None
