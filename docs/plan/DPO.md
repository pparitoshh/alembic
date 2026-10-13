# Plan: DPO on a scaled B7 student

Status: plan, 2026-10-13. Nothing in this plan is submitted. Branch `feat/data-hpc10k`. Results go to
`docs/TRAINING_PROGRESS.md` and `docs/MENTOR_REPORT_v2.md`.

## Goal

The student model must win against the base model on the v3 test set (win rate more than 0.5, mean of
3 seeds). It must not become worse on the v2 test set. DPO must remove invented flags and missing flags.

## Why

| Model | v2 win rate | v3 win rate | Seeds |
|---|---|---|---|
| B7, 100% of the change | 0.398 | 0.438 | 3 |
| B7, 70% of the change | 0.409 | 0.418 | 1 |
| B7, 30% of the change | 0.439 | 0.526 | 1 (v3), 2 (v2) |
| B7, 20% of the change | 0.477 | 0.533 | 1 |
| Teacher model (Qwen3-32B-AWQ) | 0.689 | 0.585 | – |

1. The full fine-tuning removes too much knowledge of the base model. A smaller part of the change wins.
2. The teacher model wins. Thus, there is more quality to get from the teacher model.
3. The student model still invents flags (B7 bad-flag rate on v2: 0.085; base model: 0.025).
   SFT only shows correct answers. DPO also shows which answers are wrong.

## Decision 1: the start model

DPO starts from the best scaled model, not from B7 at 100%.

1. Wait for the scaling results of all 3 seeds (judge job 60281973).
2. Select the scale s with the highest mean v3 win rate (probably 0.2 or 0.3).
3. For each seed, save the merged weights `base + s × (B7 − base)` in bf16.
   These weights are the start model and the DPO reference model.

If the other seeds do not confirm the scaling result, use B8a or B7 at 100% as the start model.

## Decision 2: the data

| Item | Plan |
|---|---|
| Questions | Prose training questions from `runs/b7_hpc10k/verified.jsonl` (the decontam step already removed the test questions and the v3 source parts) |
| Preferred answer | The verified teacher answer. Skip the question if this answer has an invented flag. |
| Rejected answer | An answer from the start model (temperature 0.8, 2 samples). Use it if it has an invented flag, or if it does not contain at least half of the teacher answer flags (minimum 2 flags). |
| Quantity | 6,000 to 12,000 questions for each seed. Target: at least 800 pairs for each seed. |
| Output | `runs/dpo_<start>/seed_<n>/pairs.jsonl` |

The first mining run (job 60284208) used B7 at 100% and found 858 pairs from 6,000 questions:
303 with invented flags and 555 with missing flags. DPO is better when the rejected answers come from the
model that we train. Thus, mine again from each scaled start model.

If there are not enough pairs, add judge pairs: the judge compares the student answer with the teacher answer,
and a clear teacher win gives a pair.

## Decision 3: the training

| Setting | Value | Reason |
|---|---|---|
| Trainer | TRL `DPOTrainer` (TRL 1.14) | Already in the project environment |
| Adapter | New LoRA, rank 16, alpha 32, all linear layers | The reference model is the start model without the new adapter. No second model copy is necessary. |
| Weights | bf16, no 4-bit | Project default from 2026-10-13 |
| beta | 0.1 | Standard value |
| Learning rate | 5e-6, cosine, 3% warmup | Small, to keep the base knowledge |
| Epochs | 1 | |
| Batch | 2, gradient accumulation 8 (effective batch 16) | |
| Maximum length | 2,048 tokens | Same as SFT |
| Loss | Standard DPO (sigmoid) | |

## Decision 4: the evaluation

1. Merge the DPO adapter into the start model.
2. Generate answers for v2 and v3: greedy, 512 new tokens, same system prompt as all other runs.
3. Judge against the base model answers with the same protocol (gpt-oss-20b, temperature 0,
   `paired-orders-v2-invalid-incomplete`).
4. Compare 3 models for each seed: base model, start model, start model + DPO.
   The difference between the start model and the DPO model is the effect of DPO.
5. Also record: check pass rate, bad-flag rate and median answer length.

## Jobs

All jobs use 1 node (4 GPUs, 480 GB) with `fan_out_seeds`: 1 GPU for each seed, 3 seeds at the same time.
On on-site days, submit with `SBATCH_RESERVATION=s_tra_ncc`.

| # | Job | What it does | Time (estimate) |
|---|---|---|---|
| 1 | `dpo_prepare` | Merge the scaled start model; start vLLM on it; mine the pairs | 15 minutes |
| 2 | `dpo_train` | DPO training with a new LoRA adapter | 30 minutes |
| 3 | `dpo_eval` | Merge the DPO adapter; answers for v2 and v3; judge against the base model | 45 minutes |

Total: approximately 1.5 hours after the scaling results are complete.

## Code to write

| File | Contents |
|---|---|
| `src/distillkit/dpo.py` | Mining (from `$WORK/alembic/jobs/dpo_mine.py`, flag fix included) and training (`DPOTrainer`) |
| `src/distillkit/merge.py` | Scaled merge: `base + s × (adapter model − base)`, saved in bf16 (from `scaled_answers.py`) |
| `distillkit` CLI | Commands `dpo-mine`, `dpo-train`, `merge` |
| `configs/qwen3_4b_dpo.yaml` | DPO settings (a new `dpo` section in `Config`) |
| `slurm/dpo_prepare.sbatch`, `slurm/dpo_train.sbatch`, `slurm/dpo_eval.sbatch` | Whole-node seed jobs |
| `tests/test_dpo.py` | Hermetic tests: pair rules, flag matching (flag names without `--`), config |

The answer and judge steps reuse `distillkit answer` and `distillkit evaluate`, with the merged model as
`student.model`. The base model answers come from the existing cache.

## Risks

| Risk | Action |
|---|---|
| Teacher answers are short (median 442 characters). DPO can make answers shorter again. | Measure the median answer length. If it decreases, prefer teacher answers that have more than 1 method. |
| Few pairs give a small effect. | Mine more questions or add judge pairs. |
| The scaling result does not hold for all seeds. | Use B8a or B7 at 100% as the start model. |
| DPO decreases the v2 result. | v2 is a check. Reject a DPO model that is worse on v2 than its start model by more than the margin. |
| The reservation ends at 18:00. | Without the reservation, the jobs wait in the normal queue. |

## Success criteria

1. Mean v3 win rate of the DPO model, 3 seeds: more than 0.5.
2. Mean v2 win rate: not lower than the start model by more than 0.05.
3. Bad-flag rate on v3: not higher than the base model (0.021).
