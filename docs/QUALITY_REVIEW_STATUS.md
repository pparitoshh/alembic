# Quality review and training status

As of 2026-10-10 15:02 UTC, **4,775/30,000** distinct Qwen teacher Q&A are accepted under the automated gates. The corrected certificate supersedes an earlier 5,319-row report after a second model audit and quarantine of two affected capabilities. The 10,000-row mark is an intermediate milestone; training on the final 30,000-row release has not yet run.

| Area | Current result | Boundary |
|---|---|---|
| Evaluation | 151 source-grounded tasks: 106 normal development, 23 first-response tool development, 22 frozen final | Tool tasks measure the first decision and arguments, not full multi-turn success. |
| Certified teacher data | [4,775 released Q&A](../data/teacher/verified_qwen_4775/README.md): 1,100 earlier accepted rows plus 3,675 from a 6,000-question new-document batch | 2,325 batch answers were withheld. The release file SHA-256 is `3f99eefb8370530434245e08faa7d7c6ca85a02438611f1350d634149b02c8e3`. |
| Active next cohort | 5,307 source-bound questions selected from 6,915 pinned new-document proposals; Qwen answer job `59887935` was running at this check | Questions and generated answers do not count until independent review, deduplication and loader certification pass. |
| Student training | An eight-record QDoRA sanity run passed in [PR #10](https://github.com/pparitoshh/alembic/pull/10). An interim Qwen3-4B QDoRA run on the 1,100-row release finished successfully on 2026-10-10: 207 optimizer steps and a saved adapter. | The 30,000-row run is held until a hash-validated certificate reaches the target. The interim adapter has no held-out improvement result yet. |

The corrected [certification report](../data/teacher/verified_qwen_4775/certification_report.json) records Qwen3-32B-AWQ as teacher, Gemma 4 26B-A4B IT as the independent LLM gate, a second GPT-OSS-20B high-risk review, and 3,675 new rows checked through the real TRL loader. Known Dask scheduling-overhead and RAJA kernel-policy false approvals were quarantined. LLM reviewers can still miss factual errors; lexical and model novelty gates cannot prove the absence of every semantic duplicate. The newer 3,675 rows carry pinned source origins in-row; the earlier 1,100 rows retain their original private-campaign source manifests and hashes but do not all carry a URL in-row.

The pipeline uses parallel independent source and question reviews where useful, then evaluates answer cohorts against the cumulative accepted bank before promoting a new certificate. Each cohort is a candidate pool, not a promised acceptance count. Scheduling depends on live CINECA account allocation, Slurm policy and storage availability; no self-imposed campaign compute cap applies.

The interim run's private `interim-training-1100-v1/training_evidence.json` reports `PASS`, 1,100 loader-checked rows, 150,795 supervised tokens, 207 optimizer steps, and adapter SHA-256 `c96eb213cb17bf062d5f415cbbc7142584de9385e12d991820fc9a45ddee8072`. It ran from 09:09:23 to 09:22:11 UTC. This establishes training execution and adapter creation, not improvement over the base model.

Next: complete the running answer cohort, independently review and certify it, acquire additional licensed advanced-document families, and continue source-bound batches to 30,000. Full-release QDoRA training and held-out evaluation follow the certified target. A complete outcome claim requires the measured student result, not a queued job or an interim adapter.
