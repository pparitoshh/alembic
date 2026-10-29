# Quality review and training status

**2026-10-11 — local corrective review. Broad-topic generation and milestone
training are paused.** The prior row-count milestone did not establish that the
data fits our Slurm/HPC assistant. See the [full workflow audit](audits/20261011_hpc_scope/README.md).

| Area | Verified state | Practical limit |
|---|---|---|
| Published data | 14,304 rows, unique IDs, no normalized exact-question duplicates or empty final answers in the independent profile | Not established as a primary Slurm training set. Preserve as possible secondary material. |
| Campaign data | 14,967 rows had cleared the recorded automated acceptance path before the pause; 663 more than the published snapshot | This is a pipeline count, not a relevance-approved count. No new data is being pushed. |
| Tool use | 10 tool-decision rows in the published set: call 3, clarify 2, deliberate no-call 5 | 0.070% versus the configured roughly 30% target. |
| Relevance | Broad libraries dominate; only 28 rows mention Slurm or common Slurm commands in the question/final answer | Keyword matching does not determine usefulness. The exact suitable subset remains unknown. |
| Six-run training | Two learning rates, seeds 42/43/44, same 12,092 rows; all completed | Correct training execution did not produce a better model. |
| Development comparison | Mean judge win rates about 30–34%; static and first-response tool scores also below base on average | G2 remains unmet; data mix is a plausible contributor, not a demonstrated sole cause. |
| 11,167 pilot | Its evaluation is complete and [results are available](student_experiments_12092/results/evaluation/pilot11167/) | Single historical seed, excluded from the controlled six-run aggregate. |
| Evaluation scope | 106 normal development tasks, 23 first-response tool tasks, 22 separate final tasks | First-response tool checks do not establish full multi-turn completion; final holdout remains untouched. |

The newer cohorts used corrected independent Gemma review, GPT-OSS high-risk
checks and loader certification. The inherited 1,100 rows have earlier review
evidence; not every row passed the same newer path. No automated reviewer proves
semantic correctness, relevance or the absence of every semantic duplicate.

Published 12,092-row training evidence records assistant-only loss masks, EOS and
padding checks and no max-length boundary hit. That supports the mechanics of
the run; it does not fix the training distribution or establish model quality.

The project-readable development evaluation copy is at:

```text
/leonardo_work/EUHPC_D30_031/alembic/campaign_evidence/student-evaluations-20261011-v1
```

Checksums and project-group permission modes were verified. A teammate's login
has not yet tested access. Public summaries and adapters remain in the
[six-run release](https://github.com/pparitoshh/alembic/releases/tag/student-comparison-12092-v1).

Next: review the source-to-evaluation workflow with the team, align with the
existing Slurm corpus, establish task relevance and accepted mix, then compare
a frozen suitable training snapshot against the base. Keep further collection
paused and bulk artifacts outside Git. Local code includes an opt-in suitability
gate draft; it is not deployed to the team's checkout or enabled in old configs.
