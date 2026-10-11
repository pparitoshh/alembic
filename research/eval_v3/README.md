# Eval v3 build and comparison

This implements PR #36's trained-topic **development** evaluation. It creates no
new training dataset. The 66-question v2 release and paired-orders judge remain
unchanged. The new release must contain 200 accepted, distinct normal QAs.

## Status, 2026-10-11

- Source freeze: complete, before new candidate generation.
- Build/review/release: job `60026161`, submitted and waiting for scheduler priority.
- Base versus B6 seeds 42/1/2: array `60026318`, after successful build only.
- Training-overlap audit: CPU job `60026319`, after successful build only.
- Consolidated results: CPU job `60026845`, after comparison/audit termination;
  it writes a complete report only when all required artifacts validate.
- Accepted v3 release and comparison scores: **not yet available**.
- Original submission `60025762` was cancelled while still pending to correct
  the multi-constraint sampling plan; it produced no candidates.

The code and frozen inputs are in the isolated writable Leonardo directory:

```
/leonardo_work/EUHPC_D30_031/runs/a08trc09/eval-v3-20261011/code
```

The teammate-owned checkout `/leonardo_work/EUHPC_D30_031/alembic/alembic` is
read-only to this account. A guarded deployment attempt was refused before any
live file was written. **Apply this PR's code and holdout manifest before merging
or training B7**, or perform the merge/training in the isolated patched checkout.
Do not assume the older live `decontam` script enforces these source exclusions.
Old adapters, historical training records and v2 outputs are unchanged.

## Frozen sources and audit

`data/eval/eval_v3.holdout.json` includes original chunk IDs and exact text hashes,
plus all eight evaluation-only composite documents. `v3_sources/plan.json` binds
the freeze and all bundle bytes. Sources are Slurm `slurm-26-05-4-1`. The four
supplemental hold/mail/output blocks restore entries removed by the training
fetcher's broad array/requeue filter, using the same pinned upstream manuals.

`source_exposure_before_release.json` binds the audited inputs by SHA-256:
910 of 12,773 historical B6 rows overlap these source chunks; the eight top-up
runs contain 1,300 additional source matches across 10,248 rows. These counts
are **not** final question-contamination counts or a deduplicated merged size.
Historical exposure cannot be reversed, so no blind-test claim is made.

Future decontamination excludes exact source hashes (including aliases), matching
document/chunk IDs, whole held-out documents, and conservatively unbound records
from a partially held-out document. This is not a detector for arbitrary semantic
paraphrases or changed chunk boundaries. Training admission also checks the freeze
before importing GPU libraries, so stale verified inputs cannot bypass it.

## Build and gates

From the repository root, with the project environment on PATH:

```bash
python -m distillkit.eval_v3 preflight
sbatch slurm/eval_v3_build.sbatch
```

Set `REPO`, `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `HF_HOME` and `VLLM_VENV` for
your installation before submission. The isolated deployed `.venv` points to
the account's usable project environment. Job scripts explicitly use this
checkout's `src` directory, rather than an older installed source tree.

The build performs:

1. Qwen3-32B-AWQ generates 512 candidates from eight source bundles. A bound,
   evaluation-only sampling plan scopes topics and supplies 64 script candidates
   for multi-constraint coverage and 64 diagnostic candidates for debugging.
2. Gemma independently reviews each answer against its exact source through the
   existing `source-support-v3` protocol; raw reports and bindings are retained.
3. The standard verifier applies deterministic syntax/flag checks, duplicate
   filtering and support gates. Its existing flag checker covers long options in
   sbatch/srun/salloc code; broader command semantics still rely on source review.
4. A separate Gemma review checks question context, premises, assessability,
   answer completeness and actual primary topic. Negative verdicts are not
   rerolled. Missing, truncated or stale reports cannot approve candidates.
5. A deterministic selection enforces the 30/30/25/25/30/20/20/20 topic quotas,
   cycles task types with how-to/debug emphasis, and checks exact, normalized
   and word-trigram near duplicates against all existing evaluation questions.
   It never reads student answers or scores.
6. Only a full accepted set writes `eval_v3.jsonl`, its manifest and the complete
   evidence sidecar. Every reference cites its source bundle chunk, with original
   source chunks resolved in the sidecar and freeze.

All reviews are model reviews; no human review is claimed. These gates reduce
errors but do not establish semantic correctness with certainty. Normal QA v3
does not replace the existing, separate tool-use evaluation.

If any topic quota is short, the build fails closed and writes
`runs/eval_v3_generation/release_status.json`. No benchmark is released and
dependent GPU evaluations are cancelled. Inspect deficits and failures before
preparing a targeted continuation; do not lower the gates or label candidates
as accepted. Generation bindings forbid changing inputs in a resumed run.

The source freeze is immutable. `prepare.py` documents its construction from the
original focus-job source snapshot and refuses to overwrite it. Ordinary users
should use the committed sources, not refetch or re-chunk them.

## Audit and evaluation

```bash
# Run after the build succeeds, with TRAIN_RUNS and B6_RUN set appropriately.
sbatch --dependency=afterok:BUILD_ID --kill-on-invalid-dep=yes slurm/eval_v3_audit.sbatch
sbatch --dependency=afterok:BUILD_ID --kill-on-invalid-dep=yes slurm/eval_v3_compare.sbatch
```

The audit reads hpc10k, all eight top-ups and historical B6 inputs. It writes new
`runs/eval_v3_audit/<index>-<run>/clean.jsonl` copies and excluded-row evidence,
plus source, question and union counts in `audit.json`. Never replace historical
B6 training data or claim the historical model was trained on these new copies.
The future merge must still deduplicate across the selected clean input files.

The comparison reads existing B6 adapters through symlinks in fresh
`runs/eval_v3_b6/seed_<seed>` directories. It verifies adapter hashes before/after,
uses the existing student prompt and greedy 512-token decoding, then judges with
gpt-oss-20b at temperature zero in both orders. Any invalid verdict leaves the
aggregate incomplete. Existing v2 caches and outputs are never reused or written.

After all jobs finish:

```bash
python -m distillkit.eval_v3 check-release
python research/eval_v3/report.py --v2-root /path/to/runs/b6_hpc10k
```

The report checks IDs, exact questions/references, paired verdict completeness,
protocol and agreement of the headline score with per-question scores. It writes
`runs/eval_v3_b6/comparison.json` and `.md`, showing v2 and v3 separately, plus v3
topic scores. Publish completed results in `docs/TRAINING_PROGRESS.md`, the final
audit counts in `data/eval/README.md`, and commit the release/evidence to PR #36.
No score, acceptance count or quality improvement should be claimed before those
artifacts exist. Shared sources and repeated seeds are correlated; the original
±0.07 figure is an independent-item approximation, not a validated confidence bound.
