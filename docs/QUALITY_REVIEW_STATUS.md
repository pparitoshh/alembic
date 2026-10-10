# Quality review and training status

As of 2026-10-10 19:03 UTC, **8,450/30,000** distinct Qwen teacher Q&A are accepted under the automated gates. The 10,000-row mark is an intermediate milestone; training on the final 30,000-row release has not yet run.

| Area | Current result | Boundary |
|---|---|---|
| Evaluation | 151 source-grounded tasks: 106 normal development, 23 first-response tool development, 22 frozen final | Tool tasks measure the first decision and arguments, not full multi-turn success. |
| Certified teacher data | [8,450 released Q&A](../data/teacher/verified_qwen_8450/README.md): the prior 4,775-row release plus 3,675 newly certified from 5,307 source-bound questions | 1,632 second-batch answers were withheld. The release SHA-256 is `0f29b8b260aafd12bb1859e4727fb494502716609742f70e50785ddd84228a9f`. |
| Active next cohort | A third Qwen answer batch completed 2,097 raw answers; independent gate job `59887941` is running | Raw answers are not verified Q&A until the corrected certificate passes. Twenty-five queued cohorts span 69,862 distinct source-section proposals; proposals are not acceptance commitments. |
| Student training | An eight-record QDoRA sanity run passed in [PR #10](https://github.com/pparitoshh/alembic/pull/10). An interim Qwen3-4B QDoRA run on the 1,100-row release finished successfully on 2026-10-10: 207 optimizer steps and a saved adapter. | The 30,000-row run is held until a hash-validated certificate reaches the target. The interim adapter has no held-out improvement result yet. |

The second [corrected certification report](../data/teacher/verified_qwen_8450/certification_report.json) records Qwen3-32B-AWQ as teacher, Gemma 4 26B-A4B IT as the independent LLM gate, a GPT-OSS-20B high-risk audit, and 3,675 new rows checked through the real TRL loader. It rejected another 397 provisional passes. The first 4,775 rows remain byte-identical to the [prior release](../data/teacher/verified_qwen_4775/README.md), whose known Dask and RAJA false approvals were quarantined. LLM reviewers can still miss factual errors; lexical and model novelty gates cannot prove the absence of every semantic duplicate. The newer 7,350 rows carry pinned source origins in-row; the earlier 1,100 rows retain private-campaign source manifests and hashes but do not all carry a URL in-row.

The pipeline uses parallel independent source and question reviews where useful, then evaluates answer cohorts against the cumulative accepted bank before promoting a new certificate. Each cohort is a candidate pool, not a promised acceptance count. Scheduling depends on live CINECA account allocation, Slurm policy and storage availability; no self-imposed campaign compute cap applies.

The interim run's private `interim-training-1100-v1/training_evidence.json` reports `PASS`, 1,100 loader-checked rows, 150,795 supervised tokens, 207 optimizer steps, and adapter SHA-256 `c96eb213cb17bf062d5f415cbbc7142584de9385e12d991820fc9a45ddee8072`. It ran from 09:09:23 to 09:22:11 UTC. This establishes training execution and adapter creation, not improvement over the base model.

Next: finish the third cohort's independent review and corrected certificate, then continue source-bound batches to 30,000. Full-release QDoRA training and held-out evaluation follow the certified target. A complete outcome claim requires the measured student result, not a queued job or an interim adapter.
