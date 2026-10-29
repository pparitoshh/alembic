# HPC data and workflow review — 2026-10-11

**The broad dataset is not established as suitable primary training data for the
Slurm assistant.** Preserve it as a possible secondary corpus. Bulk generation
and count-triggered training are paused. The next step is to review the existing
workflow with the team, not to collect another large corpus.

This is a local corrective review, not a newly published dataset or a claim that
the remaining goals are complete. The team's `feat/data-hpc10k` work is separate;
its checkout, working files and jobs have not been changed.

## Reproducible findings

The [14,304-row profile](profile_14304.json) is bound to file SHA-256
`c31c89a155d8b2cab46dd1f63387a120e2c542dec5c337622cdcba79c86af61d`.
The [12,092-row profile](profile_12092.json) covers the completed training snapshot.

| Measure | 14,304-row result | Interpretation |
|---|---:|---|
| Unique IDs | 14,304 | No repeated IDs |
| Duplicate normalized questions | 0 | NFKC, casefold and whitespace normalization only; not semantic deduplication |
| Empty final answers | 0 | Presence does not establish correctness |
| Source documents | 4,779 | Diversity alone does not establish relevance |
| Tool-decision examples | 10 (0.070%) | Call 3, clarify 2, deliberate no-call 5; far below the configured ~30% goal |
| Prose examples | 14,294 | Acceptance preserved an overwhelmingly prose dataset |
| Tasks | Concept 7,733; script 5,720; debug 572; how-to 279 | How-to/debug are underrepresented; the goal did not specify exact task quotas |
| Persona `technical practitioner` | 13,204 | Generic metadata dominates |
| Median final answer | 414 characters | Length is descriptive, not a quality score |
| Slurm/command keywords in questions | 14 | Discovery proxy only |
| Slurm/command keywords in questions or final answers | 28 | Not a count of useful rows; CUDA/MPI/profiling may be relevant without these words |

Keyword regex: whole words, case-insensitive,
`slurm|sbatch|squeue|sacct|srun|scancel|scontrol|salloc`. Different fields or matching
rules can explain different keyword counts in other audits. A defensible useful-row
count still requires a task-relevance review. No claim is made that the rest is
useless or that all keyword hits are useful.

Reproduce from the repository root with the project Python environment:

```sh
python -m distillkit.data_profile data/teacher/verified_qwen_14304/verified.jsonl --output runs/audit/profile_14304.json
python -m distillkit.data_profile data/teacher/verified_qwen_12092/verified.jsonl --output runs/audit/profile_12092.json
```

## End-to-end flow and the gaps

| Stage | Existing evidence or behavior | Required correction / boundary |
|---|---|---|
| Define task and split sources | GOAL specifies a Slurm/CUDA/MPI/profiling assistant; evaluation has separate development and final document splits | Let task coverage select the training sources. Newness, project count and advanced terminology are not relevance criteria. Keep frozen final documents excluded. |
| Select documents and questions | Advanced cohorts included source provenance/notice checks and independent question reviews | Broad-library expansion was not constrained to the assistant's workflows. Review relevance to job submission, scheduling, debugging, resource use and associated HPC development before admission. |
| Generate teacher answers | Qwen teacher produces source-bound records; most expansion rows are plain prose | Tool fraction must be measured after rejection/merging, not inferred from a generation config. Plan call, clarify and no-call scenarios explicitly. |
| Verify and accept | Corrected newer cohorts have independent Gemma review, GPT-OSS high-risk checks, global deduplication and loader evidence | Grounding and structural acceptance do not prove task fit. The inherited 1,100 rows have historical checks; do not claim every row received the newer two-model path. LLM gates can make mistakes. |
| Freeze a training release | Certificates and hashes select a reproducible snapshot | Previous milestone promotion was driven by count. Require relevance, accepted mix and existing provenance/decontamination evidence for the selected snapshot. |
| Train | Six QDoRA runs: two learning rates and seeds 42/43/44 on the same 12,092 rows | These are real completed runs, not proof of improvement. Published loader evidence records 12,092 checked rows, assistant-only masking, EOS supervision, padding masking and no max-length boundary hit (max 1,520 / 2,048 tokens). |
| Evaluate | Both judges scored cached development answers; six-run report includes mean and sample spread | Students underperformed the base. Tool eval measures first response, not full multi-turn completion. The wrong mix is a plausible contributor, not a proven sole cause. |
| Export and release | Adapters and comparison evidence exist | A successful training job does not establish G2, laptop performance, GGUF/Ollama behavior or the final release goals. Verify each separately. |

## Completed evaluation access

The six-run [comparison report](../../student_experiments_12092/RESULTS.md) found
mean judge win rates of about 30–34% versus the base. G2 is unmet.

The **11,167-row pilot evaluation is also complete**, not still queued:
[pilot summaries](../../student_experiments_12092/results/evaluation/pilot11167/).
Its GPT-OSS/Gemma judge win rates are 31.84%/31.60%. Static-check pass rate is
97.17% (base 94.34%), while tool first-response pass rate is 91.30% (base 95.65%).
This single historical seed is not part of the six-run aggregate.

Private campaign parent directories were mode 0700. Instead of widening access
to the whole campaign, an evidence-only copy now exists at:

```text
/leonardo_work/EUHPC_D30_031/alembic/campaign_evidence/student-evaluations-20261011-v1
```

The copy has 101 files, including SHA256SUMS, cached development answers, judge
outputs, calibration and summaries for all seven evaluations. Checksums were
verified; directories are 2750 and files 0640 with project group EUHPC_D30_031.
No final holdout, training dataset or adapter was copied. Access with another
team member's login has not been tested.

Public adapter release:
https://github.com/pparitoshh/alembic/releases/tag/student-comparison-12092-v1

## Local safeguard draft

`train.data_requirements` adds an opt-in check before tokenizer/model loading for
both ordinary verified-file and multi-origin-manifest training. It checks the
accepted tool fraction, presence of required decision modes/task types, basic
record structure, and a hash-bound domain review of the exact selected records.
Default bounds when enabled are 25–35% tool decisions around the existing 30%
target; these are proposed tolerances, not newly agreed team quotas. Presence
checks do not ensure a balanced distribution across tasks or personas.

The domain review uses schema `training-domain-review-v1`: the ordered canonical
`records_sha256` from the profile, decision `approved-for-hpc-training`, exact
`approved_record_ids`, a `review_method`, and an `evidence` path/SHA-256 binding.
Evidence paths are relative to the review JSON. The training configuration pins
the review JSON using `train.data_requirements.domain_review.path` and `.sha256`.
Automated LLM review can supply these decisions; manual review of every training
row is not required. Hashes prove which judgments were used, not that they are
correct. No domain-approval file has been fabricated for the broad corpus.

This draft **is not enabled in historical configurations or deployed to the
team's checkout**. Existing provenance, licensing, semantic review, tool checking
and decontamination remain required. The campaign is separately paused, with
entrypoint guards to stop the old generation/training wrappers being restarted
accidentally. The two completed-run auto-publication LaunchAgents were also disabled
and unloaded so the local review cannot trigger a new automatic publication.
All data and completed model artifacts are retained.

Validation: 94 targeted tests passed; 8 optional-dependency tests were skipped.
The remote stage entrypoint was invoked while paused and exited before launching
work. The account queue was empty at that check. No GPU training was run for this
review, and no code or data was pushed.

## Repository cleanup review

At the start of this review, tracked files totaled 104,341,696 bytes, of which
100,278,570 bytes were under `data/teacher`. Five cumulative JSONL snapshots
accounted for most of that size. This is avoidable growth, not evidence that a
hosting size limit has already been reached.

Keep future raw datasets, checkpoints and full logs in shared artifact storage;
commit small manifests with location, row count, hashes, source scope and review
results. Existing snapshot validators depend on predecessor files, so moving
them requires updating those dependencies and checking retrieval first. Deleting
working-tree files does not remove Git history. No files were deleted, public
history rewritten, new data pushed or cleanup committed in this review.

## Next decisions, in order

1. Review this flow and the existing changes with the team. Confirm ownership and
   gaps in the team's ongoing Slurm corpus before collecting anything else.
2. Use automated task-relevance review to identify a suitable subset and record
   primary versus optional secondary content. Report an actual accepted mix.
3. Freeze one source-disjoint, provenance-checked training snapshot and compare
   it with the base using the same development protocol. Keep the final holdout
   untouched. Use measured failures to decide any targeted additional data.
4. Complete the remaining GOAL must-haves: reproducible HPC pipeline, demonstrated
   quality improvement, quantization/Ollama plus measured laptop behavior, and
   an honest release/model card. Volume alone does not complete these goals.
