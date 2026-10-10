# Certified Qwen teacher Q&A: 11,167 rows

This immutable snapshot contains 11,167 distinct Q&A accepted under the campaign's automated gates as of 2026-10-10 22:09 UTC. It crosses the 10,000 milestone and is an intermediate release toward 30,000.

- `verified.jsonl`: the exact cumulative release, SHA-256 `454a9498377b344d5e6a85d4ac769fe5eb3e7d11bbf4c3fe5af52b9ef81b75aa`.
- `certification_report_third.json`, `certification_report_fourth.json`, and `certification_report.json`: corrected certificates for the three cohorts since the 8,450-row release. They add 1,286, 249, and 1,182 Q&A respectively.
- `validate.py`: checks each corrected certificate and cumulative prefix hash, the exact 8,450-row predecessor prefix, unique IDs/questions/answers, teacher and conversation shape, and pinned source URLs and evidence hashes for all 2,717 new rows.

Run `python3 data/teacher/verified_qwen_11167/validate.py`. Do not edit the release in place; a changed byte requires a new versioned certificate.

Qwen3-32B-AWQ generated the answers. Independent Gemma 4 source review, a GPT-OSS-20B audit of higher-risk answers, provenance and deduplication checks, and the real TRL loader determined admission. The new cohorts draw on pinned Apache Beam, TensorFlow, PETSc, PyTorch, Triton, vLLM, Hadoop, and XLA documentation. Their accepted rows carry source URLs and hashes. The first 8,450 rows are byte-identical to the [previous certified release](../verified_qwen_8450/README.md).

A separate Qwen3-4B 4-bit QDoRA run was triggered by this milestone on Leonardo; its training and evaluation outcomes are recorded separately. Automated certification is not exhaustive human review or a guarantee against every semantic duplicate or factual mistake. The release remains overwhelmingly prose and does not yet meet the planned tool-trace mix. Source documentation is referenced by URL and hash, not redistributed here. Keep held-out evaluation families separate from training.
