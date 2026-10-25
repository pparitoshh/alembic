# Certified Qwen teacher Q&A: 12,092 rows

Immutable cumulative release certified on 2026-10-10 at 23:49 UTC. This adds 925 Q&A to the byte-identical [11,167-row predecessor](../verified_qwen_11167/README.md): 232 from cohort six and 693 from cohort seven.

SHA-256 of `verified.jsonl`: `0d26fd2ec8fa5fb5221d22aa6249f6d107201d9f4ae0b06da4734fd105f7fd9c`.

Run `python3 data/teacher/verified_qwen_12092/validate.py` to verify certificate hashes, cumulative prefixes, unique IDs and normalized questions/answers, teacher identity, and pinned source provenance for the new rows.

Qwen3-32B-AWQ generated the answers. Admission uses independent Gemma source review, a GPT-OSS audit of higher-risk answers, deduplication and provenance checks, and the real TRL loader. Automated review can miss factual errors and semantic duplicates. The data remains predominantly prose; tool-calling quality and held-out student improvement have not been established.

The completed student used the earlier 11,167-row snapshot. A separate full-snapshot QDoRA run on these 12,092 rows was submitted as Leonardo job `59940941`; a queued run is not a completed adapter. The campaign continues toward the published 50,000-example milestone.
