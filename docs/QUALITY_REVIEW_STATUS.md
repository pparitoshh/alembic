# Quality review and training status

As of 2026-10-10 15:02 UTC, **4,775/30,000** distinct Qwen teacher Q&A are accepted under the automated gates. The corrected certificate supersedes an earlier 5,319-row report after a second model audit and quarantine of two affected capabilities. The 10,000-row mark is an intermediate milestone; production student training has not yet run.

| Area | Current result | Boundary |
|---|---|---|
| Evaluation | 151 source-grounded tasks: 106 normal development, 23 first-response tool development, 22 frozen final | Tool tasks measure the first decision and arguments, not full multi-turn success. |
| Certified teacher data | [4,775 released Q&A](../data/teacher/verified_qwen_4775/README.md): 1,100 earlier accepted rows plus 3,675 from a 6,000-question new-document batch | 2,325 batch answers were withheld. The release file SHA-256 is `3f99eefb8370530434245e08faa7d7c6ca85a02438611f1350d634149b02c8e3`. |
| Active next cohort | 5,307 source-bound questions selected from 6,915 pinned new-document proposals; Qwen answer job `59887935` was running at this check | Questions and generated answers do not count until independent review, deduplication and loader certification pass. |
| Student training | An eight-record QDoRA sanity run passed in [PR #10](https://github.com/pparitoshh/alembic/pull/10) | The 30,000-row production run is held until a hash-validated certificate reaches the target. No student improvement has been measured. |

The corrected [certification report](../data/teacher/verified_qwen_4775/certification_report.json) records Qwen3-32B-AWQ as teacher, Gemma 4 26B-A4B IT as the independent LLM gate, a second GPT-OSS-20B high-risk review, and 3,675 new rows checked through the real TRL loader. Known Dask scheduling-overhead and RAJA kernel-policy false approvals were quarantined. LLM reviewers can still miss factual errors; lexical and model novelty gates cannot prove the absence of every semantic duplicate. The newer 3,675 rows carry pinned source origins in-row; the earlier 1,100 rows retain their original private-campaign source manifests and hashes but do not all carry a URL in-row.

The pipeline uses parallel independent source and question reviews where useful, then evaluates answer cohorts against the cumulative accepted bank before promoting a new certificate. Each cohort is a candidate pool, not a promised acceptance count. Scheduling depends on live CINECA account allocation, Slurm policy and storage availability; no self-imposed campaign compute cap applies.

Next: complete the running answer cohort, independently review and certify it, acquire additional licensed advanced-document families, and continue source-bound batches to 30,000. Production QDoRA training and held-out evaluation follow the certified target. A complete training claim requires the measured student result, not a queued job or the earlier sanity run.
