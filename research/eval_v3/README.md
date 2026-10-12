# Eval v3 build and comparison

This implements PR #36's trained-topic **development** evaluation. It creates no
new training dataset. The 66-question v2 release and paired-orders judge remain
unchanged. The new release must contain 200 accepted, distinct normal QAs.

## Status, 2026-10-12

- Source freeze: complete; v2 is unchanged.
- Initial build `60026161`: ran for 46m22s, generated 512/512 candidates;
  144 passed source review, 23 were rejected and 345 stayed pending. The release
  failed closed; dependent comparisons/audit were cancelled.
- Root causes: the quality gate treated the literal uncertainty value `none` as
  uncertainty, and many support reviews copied invalid evidence/unit IDs.
- Corrected CPU replay of the preserved reviews: **134 quality-eligible candidates,
  111 selected toward the fixed topic quotas; no 200-row release yet**.
- Format-repair job **`60045425` is running**, reusing existing candidates in
  `runs/eval_v3_repair_20261012`. It may retry 292 structurally invalid reviews;
  53 pending reviews with negative/uncertain judgments remain excluded. Original
  generation and review evidence stays in `runs/eval_v3_generation`.
- Replacement base/B6 array `60045495` and audit `60045496` depend on repair/release
  success. Report job `60045542` follows their termination. The original report
  `60026845` correctly recorded an incomplete result rather than inventing scores.
- Accepted v3 release and new model scores: **not yet available**.
- Original submission `60025762` was cancelled while pending to correct sampling;
  it produced no candidates.

The quality-gate fix accepts only an empty uncertainty string or the exact `none`
sentinel (ignoring case/outer whitespace). `uncertain`, qualifications such as
`none except ...`, and unrecognized values still block acceptance. Fresh review
responses use an enum for this field. The raw reviews are not edited or relabelled.

The support retry constrains transport IDs to actual packet IDs, keeps the existing
`source-support-v3` prompt/validator, and allows at most two attempts per candidate.
It does not retry explicit unsupported/uncertain verdicts, even when those reports
also contain structural errors. Every before/after report and validation result
is retained in `support_retry_history.jsonl`. The packet, source, unit-coverage and
chronology checks remain mandatory. Run `slurm/eval_v3_retry.sbatch` only against
an isolated copy of the original run; it never regenerates questions or answers.

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
