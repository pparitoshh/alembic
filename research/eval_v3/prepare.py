"""Freeze eval-only source bundles before generating any candidates.

Run from the repository root. Input is the exact focus-job seed snapshot, not
the current live website. Supplementary option blocks are pinned upstream.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import yaml
from distillkit.fetch_sources import roff_to_markdown
from distillkit.seeds import chunk_text

TOPICS = {
    "dependencies": (30, {"slurm_man_sbatch": [0, 17, 18, 19, 37, 57]}),
    "accounting": (30, {"slurm_man_sacct": [0, 2, 5, 6, 11, 23, 24, 27], "slurm_guide_job_exit_code": [0, 1, 2]}),
    "job_control": (25, {"slurm_man_scontrol": [9, 18, 21, 32, 37], "slurm_man_sbatch": [65]}),
    "output_mail": (25, {"slurm_man_sbatch": [1, 37, 39, 55, 71, 72]}),
    "resources": (30, {"slurm_man_sbatch": [16, 32, 33, 34, 40, 44, 46, 51, 52, 54]}),
    "queue_cancel": (20, {"slurm_man_squeue": [0, 1, 17, 21, 23, 25], "slurm_man_scancel": [0, 1]}),
    "multi_constraint": (20, {"slurm_man_sbatch": [0, 2, 16, 40, 51, 52, 54, 64, 65]}),
    "debug": (20, {"slurm_guide_faq": [19, 20, 21, 25, 26], "slurm_man_sacct": [11, 23, 24], "slurm_guide_job_exit_code": [0, 1]}),
}


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, default=Path("runs/eval_v3_build/original_seeds"))
    args = ap.parse_args()
    out = Path("data/eval/v3_sources")
    out.mkdir(parents=True, exist_ok=True)
    freeze = Path("data/eval/eval_v3.holdout.json")
    if freeze.exists():
        raise ValueError("holdout already frozen; do not silently regenerate it")
    origins = {d["doc_id"]: d for d in json.loads((args.seeds / "sources_manifest.json").read_text())["documents"]}
    sources, bundles = {}, []
    for topic, (target, mapping) in TOPICS.items():
        parts, keys = [], []
        for doc, indices in mapping.items():
            raw = (args.seeds / f"{doc}.md").read_bytes()
            assert sha(raw) == origins[doc]["seed_sha256"], doc
            chunks = chunk_text(raw.decode(), 1500)
            for i in indices:
                text, key = chunks[i], f"{doc}#{i}"
                sources[key] = {"kind": "chunk", "doc_id": doc, "chunk_id": key,
                    "source_sha256": sha(text), "document_sha256": sha(raw), "text": text,
                    "url": origins[doc]["url"], "source_version": origins[doc]["source_version"],
                    "license": origins[doc]["license"]}
                parts.append(f"## Source excerpt: {key}\n\n{text}")
                keys.append(key)
        # The training fetcher removes entire blocks mentioning held-out topics;
        # this also removes useful hold/mail/output definitions. Recover only
        # these exact option entries from the SAME pinned upstream source.
        supplemental = {"job_control": ("scontrol", [r"hold <job_list>"]),
                        "output_mail": ("sbatch", [r".*--mail-type=.*", r".*--output=.*", r".*--error=.*"])}
        if topic in supplemental:
            command, patterns = supplemental[topic]
            url = origins[f"slurm_man_{command}"]["url"]
            raw = urllib.request.urlopen(url, timeout=30).read()
            converted = roff_to_markdown(raw.decode())
            for pattern in patterns:
                found = [(m.group(1), m.group(0).strip()) for m in re.finditer(r"(?ms)^\*\*(.*?)\*\*\n.*?(?=^\*\*|\Z)", converted) if re.fullmatch(pattern, m.group(1))]
                if len(found) != 1:
                    raise ValueError(f"unresolved pinned option {command}: {pattern}")
                label, text = found[0]
                key = f"upstream_{command}_{sha(label)[:12]}"
                sources[key] = {"kind": "chunk", "doc_id": f"upstream_slurm_man_{command}",
                    "chunk_id": key, "source_sha256": sha(text), "text": text, "url": url,
                    "source_version": origins[f"slurm_man_{command}"]["source_version"],
                    "upstream_sha256": sha(raw), "license": "GPL-2.0", "option": label}
                parts.append(f"## Source excerpt: {key}\n\n{text}")
                keys.append(key)
        doc_id = f"eval_v3_{topic}"
        text = f"# Evaluation source bundle: {topic.replace('_', ' ')}\n\n" + "\n\n".join(parts) + "\n"
        path = out / f"{doc_id}.md"
        path.write_text(text)
        sources[doc_id] = {"kind": "document", "doc_id": doc_id, "source_sha256": sha(text),
                          "path": str(path)}
        bundles.append({"topic": topic, "target": target, "doc_id": doc_id,
                        "path": str(path), "sha256": sha(text), "source_chunks": keys})
    now = datetime.now(timezone.utc).isoformat()
    freeze.write_text(json.dumps({"version": "eval-source-holdout-v1", "created_utc": now,
        "split": "development", "strategy": "fixed chunks of previously used source documents",
        "historical_exposure": "B6 may have consumed these source chunks. Exclusions protect FUTURE merges; historical exposure is audited, not erased.",
        "sources": list(sources.values())}, indent=2) + "\n")
    (out / "plan.json").write_text(json.dumps({"version": 1, "created_utc": now,
        "target_rows": 200, "candidate_questions": 512, "chunk_chars": 50000,
        "holdout_sha256": sha(freeze.read_bytes()), "topics": bundles}, indent=2) + "\n")
    (out / "sources_manifest.json").write_text(json.dumps({"purpose": "eval-only generation",
        "documents": bundles, "holdout_sha256": sha(freeze.read_bytes())}, indent=2) + "\n")
    cfg = yaml.safe_load(Path("configs/gen_hpc10k_focus.yaml").read_text())
    cfg["run_dir"] = "runs/eval_v3_generation"
    cfg["seeds"] = {"dir": str(out), "eval_docs": ["slurm_job_arrays", "slurm_requeue_signals"], "chunk_chars": 50000}
    cfg["generate"].update(questions_per_chunk=64, personas=cfg["generate"]["personas"][:16],
                           tool_fraction=0.0, gold_dir=None)
    cfg["teacher"].update(top_logprobs=0, max_tokens=1024)
    cfg["eval"]["tool_file"] = None
    Path("configs/gen_eval_v3.yaml").write_text("# Evaluation candidates only. Never train from this run.\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(f"Frozen {len(sources)} source exclusions; 8 bundles; 512 candidates -> target 200.")


if __name__ == "__main__":
    main()
