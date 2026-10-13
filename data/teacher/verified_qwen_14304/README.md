# Certified Qwen teacher Q&A: 14,304 rows

Immutable cumulative release certified on 2026-10-11 at 08:32 UTC. The eighth advanced-document cohort added 2,212 Q&A to the byte-identical [12,092-row predecessor](../verified_qwen_12092/README.md). Its 3,242 answer candidates produced 2,212 corrected admissions and 1,030 withheld rows.

SHA-256 of `verified.jsonl`: `c31c89a155d8b2cab46dd1f63387a120e2c542dec5c337622cdcba79c86af61d`.

Run `python3 data/teacher/verified_qwen_14304/validate.py` to check the corrected certificate, exact cumulative prefix, unique IDs and normalized questions/answers, teacher identity, and pinned source provenance.

Qwen3-32B-AWQ generated the answers. Admission used independent Gemma source review, a GPT-OSS high-risk audit, deduplication and provenance checks, and the real TRL training loader. Automated review can still miss factual errors and semantic duplicates. The data is predominantly prose; the six-run QDoRA comparison on the earlier 12,092-row snapshot [did not beat the base model](../../../docs/student_experiments_12092/RESULTS.md). No improved student is claimed for this release.

The next advanced-document cohorts continue through the same cumulative quality gate. The published milestone remains at least 50,000 verified Q&A; queued proposals do not guarantee this yield.
