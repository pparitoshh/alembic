# Eval v3 build and comparison

This implements PR #36's trained-topic **development** evaluation. It creates no
new training dataset. The 66-question v2 release and paired-orders judge remain
unchanged. The new release must contain 200 accepted, distinct normal QAs.

## Status, 2026-10-12

- **Released: 200 normal Slurm QAs**, with every topic quota filled, manifest and
  complete selected-row review evidence. The original 66-row v2 is unchanged.
- Build `60045425` and the original training-input audit `60045496` completed.
- The adapter owner granted read access to all three B6 weight files. Direct
  reads as `a08trc09` now succeed.
- Final B7 audit **`60211738` completed** against the final frozen v3 release:
  **20,472 rows, zero source matches, zero lexical question matches**, input hash
  unchanged. Evidence: `b7_final_audit.json` and `b7_final_audit_context.json`.
- The owner applied exclusions in a separate PR worktree and merged B7 there,
  without patching the older shared checkout. B7 training was already running
  when this independent final-question audit completed; this was not a claim
  that our audit preceded training.
- Comparison array `60211313` reached inference but exposed an incorrect runtime:
  the private checkout's `.venv` linked to a shared CUDA 13 build incompatible
  with the allocated nodes. It was cancelled before any answer caches were
  written. Replacement array **`60213268`** uses a private environment provisioned from
  the repository's locked CUDA 12.9 dependencies. The comparison now tests
  GPU availability before loading weights and records runtime versions.
- Comparison array **`60213268` completed** for all three seeds. V3 scores:
  seed 42 **43.125%**, seed 1 **43.375%**, seed 2 **41.500%**. Mean **42.67%**,
  with sample SD **1.02 percentage points** across seeds. B6 remains below the
  50% comparison baseline, so no overall model-quality improvement is established.
- Report **`60213647` completed** and was rerun locally. All 600 question/seed
  pairs have both judge orders, with zero invalid verdicts. Base answers are
  identical across seeds. Historical v2 results are unchanged and reported
  separately (mean 36.87%, seed SD 3.74 percentage points).
- Results: `b6_v3_comparison.md`, `b6_v3_comparison.json`, all per-question paired
  scores in `b6_v3_verdicts.jsonl`, and runtime/adapter/artifact bindings in
  `b6_v3_run_manifest.json`. The requested comparison and final B7 audit are done.

### Audit path fix

`eval_v3 audit` and `check-release` resolve the evaluation directory from the
checkout supplying the Python module (`heldout.EVAL_ROOT`). The caller's working
folder cannot silently substitute an older eval directory. Audit fails before
writing outputs if the v3 holdout is missing/empty or the complete frozen release
is unavailable or has changed. `audit_context.json` records the resolved directory,
release/holdout hashes and every question-file hash. Inputs and output paths remain
relative to the caller, as normal command-line paths.

The final B7 audit intentionally ran from the older shared checkout with this
worktree's module on `PYTHONPATH`, reproducing the reported invocation while
correctly excluding against the frozen 200-question release.

### Build history

- Initial build `60026161`: ran for 46m22s, generated 512/512 candidates;
  144 passed source review, 23 were rejected and 345 stayed pending. The release
  failed closed; dependent comparisons/audit were cancelled.
- Root causes: the quality gate treated the literal uncertainty value `none` as
  uncertainty, and many support reviews copied invalid evidence/unit IDs.
- Corrected CPU replay initially yielded 134 quality-eligible candidates and
  111 toward the fixed topic quotas. The completed repair filled the deficits.
- Format-repair job `60045425` reused existing candidates in
  `runs/eval_v3_repair_20261012`. It could retry 292 structurally invalid reviews;
  53 pending reviews with negative/uncertain judgments remain excluded. Original
  generation and review evidence stays in `runs/eval_v3_generation`.
- The original report `60026845` correctly recorded an incomplete result.
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
live file was written. The owner used a separate PR worktree to apply exclusions
before merging B7.
Future merges/training must likewise use a checkout containing the code and freeze.
Do not assume the older live `decontam` script enforces these source exclusions.
Old adapters, historical training records and v2 outputs are unchanged.

### Adapter access handoff (resolved)

The adapter owner granted read access to the three weight files. The original
least-privilege handoff command was:

```bash
for seed in 42 1 2; do
  setfacl -m u:a08trc09:r "/leonardo_work/EUHPC_D30_031/alembic/alembic/runs/b6_hpc10k/seed_${seed}/adapter/adapter_model.safetensors"
done
```

This does not grant write access or change model bytes. Directory traversal and
adapter config reads already work. Verify actual file reads as `a08trc09` before
resubmitting the comparison array; file existence alone did not establish access.
Then use the isolated patched checkout and the existing B6 adapters:

```bash
export REPO=/leonardo_work/EUHPC_D30_031/runs/a08trc09/eval-v3-20261011/code
export B6_RUN=/leonardo_work/EUHPC_D30_031/alembic/alembic/runs/b6_hpc10k
export VLLM_VENV=/leonardo_work/EUHPC_D30_031/venvs/vllm
export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache
export SBATCH_ACCOUNT=EUHPC_D30_031 SBATCH_PARTITION=boost_usr_prod
cd "$REPO"
python3 - <<'PY' && sbatch slurm/eval_v3_compare.sbatch
import os
from pathlib import Path
for seed in (42, 1, 2):
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        with (Path(os.environ['B6_RUN']) / f'seed_{seed}' / 'adapter' / name).open('rb') as f:
            assert f.read(1), (seed, name)
print('All three adapters are readable.')
PY
```

The completed release and audit do not need rerunning. After all three seeds
finish, run the report command below; it must validate all 200 paired judgments
per seed before publishing scores. The B7 source exclusions were handled
separately in the owner's PR worktree.

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

The completed post-release audit is in `decontamination_audit.json`, including
input/output hashes, source and question counts, and locations of filtered copies.
All ten inputs had zero exact/normalized/lexical-near question matches against the
evaluation files. This does not prove absence of semantic paraphrases.

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
