# Certified Qwen teacher Q&A: 8,450 rows

This immutable snapshot contains 8,450 distinct Q&A accepted under the campaign's automated gates as of 2026-10-10 18:25:57 UTC. It is an intermediate release toward 30,000; final student training has not run.

- `verified.jsonl`: the exact combined release, SHA-256 `0f29b8b260aafd12bb1859e4727fb494502716609742f70e50785ddd84228a9f`.
- `certification_report.json`: corrected certificate for the second batch; 4,775 previously certified rows plus 3,675 newly certified rows. Of 5,307 answer attempts, 1,632 were withheld, including 397 by the second high-risk model audit.
- `validate.py`: verifies the hash, count, exact 4,775-row predecessor prefix, unique IDs, questions and answers, conversation shape, and pinned source URLs for all newly accepted rows.

Run `python3 data/teacher/verified_qwen_8450/validate.py`. Do not edit the release in place; a changed byte requires a new versioned certificate.

Qwen3-32B-AWQ generated the answers. Independent Gemma 4 source review, a GPT-OSS-20B audit of higher-risk answers, provenance and deduplication checks, and the real TRL loader determined admission. The second batch adds 3,675 rows sourced from pinned documentation for LLVM, Flink, Transformers, Arrow, Numba, DeepSpeed, Accelerate and Horovod. Its accepted rows carry `source_origin` URLs and source hashes. The first 4,775 rows are byte-identical to the [previous certified release](../verified_qwen_4775/README.md).

This is automated certification, not exhaustive human review or a guarantee against every semantic duplicate or factual mistake. The release is overwhelmingly prose, so it does not yet meet the planned tool-trace mix. Source documentation is referenced by URL and hash, not redistributed here. Keep held-out evaluation families separate from training.
