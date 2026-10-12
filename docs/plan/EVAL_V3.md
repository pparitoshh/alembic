# Eval v3: a larger trained-topic eval set

Status (2026-10-12): initial build generated 512 candidates but failed release. A quality-gate `none` sentinel bug is fixed; preserved reports now yield 134 quality-eligible candidates and 111 toward topic quotas. Job `60045425` is retrying only structurally invalid support reviews. Replacement comparison `60045495`, audit `60045496` and report `60045542` are scheduled. No v3 release or new model scores exist yet. See `research/eval_v3/README.md` for evidence, safeguards and run paths.

## Why

The only normal eval set is `data/eval/eval_all_v2.jsonl` (66 questions). It has two problems:

1. **Too noisy.** The 95% interval on a 66-item win rate is about ±0.12. It shows the student is below the
   base (B6 0.37 vs 0.50) but cannot separate batches: B3a, B3b and B6 (0.36-0.37) are all inside the noise.
   One question moves the win rate by about 0.015.
2. **Too narrow.** All 66 come from two docs (`slurm_job_arrays`, `slurm_requeue_signals`), and those topics
   are held out of training (`heldout_pattern` in `data/sources/hpc10k_sources.json`). So v2 measures
   transfer to unseen topics only. It says nothing about the areas we train on.

Current results (base / B6): check pass 0.955 / 0.949, bad-flag rate 0.025 / 0.098, win rate 0.50 / 0.37.
Full table: `docs/TRAINING_PROGRESS.md` (branch `docs/training-progress`).

## Goal

About 200 new normal questions on **trained** topics, reported alongside v2:

| Set | Measures | Size |
|---|---|---|
| `eval_all_v2.jsonl` (frozen, unchanged) | Unseen topics (arrays, requeue) | 66 |
| `eval_v3.jsonl` (new) | Trained topics | ~200 (interval about ±0.07) |

Same schema as v2: `id`, `doc_id`, `task`, `question`, `reference`.

## Topic mix (B6 weak areas first)

| Area | Target |
|---|---|
| Dependencies, `--parsable` chaining | 30 |
| sacct states and exit codes | 30 |
| scontrol hold / release / update, `--time-min` | 25 |
| Output files, `--open-mode`, mail notifications | 25 |
| Resource requests (CPU, memory, GPU, nodes/tasks) | 30 |
| squeue / scancel / job states | 20 |
| Multi-constraint scripts | 20 |
| Debugging failed jobs | 20 |

Task mix roughly like v2: script, concept, howto, debug. B6 is weakest on howto (0.30) and debug (0.35).

## Rules

1. **Do not change v2** or the judge protocol (`paired-orders-v2-invalid-incomplete`, gpt-oss-20b, temperature 0).
2. **No overlap with training.** `distillkit decontam` drops training records that overlap anything under
   `data/eval/`. Run it against `runs/hpc10k/clean.jsonl` and the top-up runs once v3 lands, and report how many
   records it removes.
3. **Source family (chosen: option b, development only).** `data/eval/README.md` says fresh final testing needs separately
   held-out source families. Our top-up generation already uses the sbatch, sacct, scontrol, squeue and scancel
   man pages, so options are:
   - (a) Hold out whole source docs (e.g. another site's user guide) and keep them out of training. Cleaner;
     needs new sources.
   - (b) Hold out chunks of the same docs and drop those chunks at merge. Faster, but a dev set, not blind.
4. **References** must cite the source chunk. Every flag must exist in the Slurm version we target (use the
   verify flag gate). No invented options.
5. Review: model review plus automated validation, as for v2 (`data/eval/README.md`). Add a manifest like
   `eval_all_v2.manifest.json`.

## Steps

1. Pick option (a) or (b) above.
2. Write a generation config (start from `configs/gen_hpc10k_focus.yaml`) and run `slurm/generate_gated.sbatch`
   on the held-out sources. Over-generate (~400) and select ~200 to hit the topic mix.
3. Dedupe (exact, normalized, near-duplicate) and review.
4. Add `data/eval/eval_v3.jsonl` + manifest + README section.
5. Run base and B6 on v3 to get the first numbers. Judging time grows about 4x per seed (~10 → ~40 min).

## Done when

- `eval_v3.jsonl` with ~200 reviewed items and its manifest is merged.
- Decontam has run against the training data and its counts are in the README.
- Base and B6 have v3 numbers in `docs/TRAINING_PROGRESS.md`.

## Implementation evidence

The source freeze contains 53 existing source chunks, 4 recovered option blocks from the same pinned upstream Slurm manuals, and 8 evaluation-only composite documents. The composite documents are also excluded from training. `source_exposure_before_release.json` records 910 overlapping source rows in historical B6 and 1,300 in the eight top-ups. These are source-overlap counts, not duplicate-question counts. B6 is not a clean holdout result on these sources; v3 is for development, and future merged inputs must exclude these chunks. The final lexical audit depends on accepted v3 questions and has not run yet.

Question-level sampling errors quoted above are rough independent-item approximations. Shared source chunks, repeated seeds on the same questions, and two judge orders are correlated; 400 verdicts do not count as 400 independent evaluation questions. Small changes must not be described as established improvements.
