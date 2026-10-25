# Certified Qwen teacher Q&A: 4,775 rows

This snapshot contains 4,775 distinct teacher Q&A accepted under the campaign's **automated gates** as of 2026-10-10 14:42:42 UTC. It is an intermediate release toward the 30,000-record target, not evidence that production student training is complete.

- `verified.jsonl`: the exact combined release, one JSON object per line; SHA-256 `3f99eefb8370530434245e08faa7d7c6ca85a02438611f1350d634149b02c8e3`.
- `certification_report.json`: corrected batch certificate; 1,100 previously accepted rows plus 3,675 newly certified rows; 2,325 of 6,000 new-document candidates withheld.
- `validate.py`: checks the file hash, row count, unique IDs and normalized questions, conversation shape, and the new rows' pinned source URLs.

Run `python3 data/teacher/verified_qwen_4775/validate.py` from any directory. Do not edit `verified.jsonl` in place: a changed byte requires a new versioned certificate.

The new rows were generated with `Qwen/Qwen3-32B-AWQ`, reviewed with independent `google/gemma-4-26B-A4B-it`, and risk-selected answers received a second `openai/gpt-oss-20b` audit. New rows passed the real TRL loader checks. Known Dask scheduling-overhead and RAJA kernel-policy false approvals were quarantined. Earlier rows preserve their original accepted status and source hashes; their source URLs are not consistently present in-row, so use the campaign manifests for full lineage. The new rows carry pinned `source_origin` links.

These are automated-gate certifications, not a claim of exhaustive human review or mathematical proof of semantic uniqueness. Models can share blind spots, source material can change outside the pinned snapshots, and no live cluster command execution or student quality gain is established here. The accepted release is overwhelmingly prose (4,765 prose and ten legacy non-prose rows), so it does not yet meet the originally planned tool-trace mix.

Source documentation is referenced by pinned URL and hash; the documentation text and model weights are not redistributed here. Check individual upstream terms before republishing or adapting source-derived content under a different license. Keep held-out evaluation families separate from training.
