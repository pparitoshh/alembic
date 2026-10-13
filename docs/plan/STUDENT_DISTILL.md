# Plan: student distillation experiments on the existing B7 data

Status: plan, 2026-10-13. Nothing in this plan is submitted. Branch `feat/data-hpc10k`.
Results go to `docs/TRAINING_PROGRESS.md` and `docs/MENTOR_REPORT_v2.md` (means of 3 seeds only).

## Goal

The student model must win against the base model on the v3 test set (win rate more than 0.5, mean of
3 seeds). It must not become worse on the v2 test set by more than 0.05.

## Scope

- We change only the training method. We do not generate new training data.
- All experiments use the B7 data: `runs/b7_hpc10k/verified.jsonl` (20,472 examples, decontam complete).
- The teacher top-20 token probabilities for this data are on disk:
  `runs/hpc10k_*/teacher_logprobs.jsonl.gz` and `runs/gen_slurm_tools*/teacher_logprobs.jsonl.gz`.
- DPO (`docs/plan/DPO.md`) is a separate plan. It starts after this plan.

## Facts that the plan uses

| Fact | Value | Source |
|---|---|---|
| B7 (SFT, 4-bit, full change) | v2 0.398, v3 0.438 | 3 seeds |
| B7 with 20% of the change | v2 0.479, v3 0.522 | 2 seeds |
| B8a (lr 2e-5, rank 8, half epoch, 4-bit) | v3 0.347 | 3 seeds |
| Teacher model | v2 0.689, v3 0.585 | – |
| Median answer length: base / training data / B7 | 1,381 / 442 / 313 characters | v2 |
| Validation loss minimum | approximately half of the epoch | B6 |

Conclusion: the full SFT change removes base knowledge. A small change wins. A gentle run (B8a) does not
win. Thus, the problem is the direction of the change, not only its size.

## Experiments

All experiments train from the base model with bf16 weights (`configs/qwen3_4b_dora_bf16.yaml` as the
start). Each experiment uses 1 node: 3 seeds (42, 1, 2), 1 GPU for each seed (`fan_out_seeds`).
The experiments do not depend on each other. Thus, all can run at the same time.

| # | Name | Change from B7 | Question | Code | Train time |
|---|---|---|---|---|---|
| D0 | B9 control | bf16 weights, not 4-bit | Is bf16 SFT different from 4-bit SFT? Reference for D1 to D4. | Config only | ~1.5 h |
| D1 | LoRA | LoRA, not DoRA (bf16) | Does LoRA forget less? (Biderman et al. 2024) | Config only | ~1.2 h |
| D2 | Logit KD | Loss = 0.5 × CE + 0.5 × KL(teacher top-20 ‖ student), temperature 1 | Do the teacher probabilities give more information than the text? | New loss in `train.py` | ~1.5 h |
| D3 | KL to base | Loss = CE + λ × KL(student ‖ base), λ = 0.1; base = the model with the adapter off | Does a penalty during training keep the base knowledge (the same effect as scaling)? | New loss in `train.py` | ~2 h |
| D4 | Early checkpoints | No training: evaluate the saved B7 adapters at steps 300, 500 and 700 | Is a shorter B7 run better than the full run? | None (eval only) | 0 |

After the evaluation, apply the scaled merge (`base + s × (model − base)`, s = 0.2, 0.3, 0.5) to the best 2 of
D0 to D3. This costs only answer and judge time.

## Phase 2 (only if D2 or D3 gives a result above 0.5)

| # | Name | Change | Note |
|---|---|---|---|
| D5 | D2 + D3 | Logit KD and KL to base in one loss | Combine the 2 best changes |
| D6 | On-policy distillation (GKD) | The student writes the answers; the teacher scores each token | The teacher must run during training (2 GPUs for each seed) |

## Branches and worktrees

`feat/data-hpc10k` is the supervisor branch. It holds this plan, the shared code and the results.
Each experiment has its own branch from the supervisor branch and its own worktree.

| Experiment | Branch | Local worktree |
|---|---|---|
| D0 | `exp/d0-bf16` | `../alembic-wt/d0-bf16` |
| D1 | `exp/d1-lora` | `../alembic-wt/d1-lora` |
| D2 | `exp/d2-kd` | `../alembic-wt/d2-kd` |
| D3 | `exp/d3-basekl` | `../alembic-wt/d3-basekl` |
| D4 | none (eval only, from the supervisor branch) | – |

Rules:

1. Shared code (the `compute_loss` subclass and the config fields used by D2 and D3) goes to the
   supervisor branch first. Then each experiment branch merges the supervisor branch.
2. An experiment branch changes only its own config, its own loss code and its own tests.
3. Each worktree has its own `.venv` (`uv sync`). Run `uv run pytest -q` before a push.
4. On Leonardo, each experiment has a worktree `$WORK/alembic/wt-<name>` with the same branch.
   Each experiment writes to its own `run_dir` (`runs/d<n>_<name>`).
5. A branch merges back into the supervisor branch only after its result is in the results table.

## Code to write

| File | Contents |
|---|---|
| `src/distillkit/config.py` | New fields in `train`: `kd_alpha`, `kd_temperature`, `base_kl` (default 0; old configs do not change) |
| `src/distillkit/kd.py` | Load the teacher top-20 probabilities by row id; align them to the student tokens; KL loss on the top-20 tokens |
| `src/distillkit/train.py` | A `SFTTrainer` subclass with `compute_loss` that adds the KD term and the base-KL term |
| `configs/qwen3_4b_d1_lora.yaml`, `_d2_kd.yaml`, `_d3_basekl.yaml` | One config for each experiment (D0 uses `qwen3_4b_dora_bf16.yaml`) |
| `tests/test_kd.py` | Hermetic tests: token alignment, KL value on a small known case, loss = CE when the weights are 0 |

## Checks before the first job

1. Make sure that Qwen3-32B-AWQ and Qwen3-4B-Instruct-2507 use the same tokenizer. Logit KD needs this.
2. Make sure that each row in `verified.jsonl` has its probabilities in a `teacher_logprobs` file.
   Rows without probabilities use CE only. Record the fraction.
3. Make sure that the token ids in the probabilities file (`return_tokens_as_token_ids`) agree with the
   student tokenization of the same answer. Record the fraction of rows that agree.
4. Make sure that the B7 step 300, 500 and 700 adapters exist (`keep_adapter`).

## Evaluation

The same protocol for all experiments:

1. Answers for v2 and v3: greedy, 512 new tokens, the same system prompt.
2. Judge: gpt-oss-20b, temperature 0, `paired-orders-v2-invalid-incomplete`, against the cached base answers.
3. Record for each model: v2 and v3 win rate (mean of 3 seeds), check pass rate, bad-flag rate,
   median answer length.

## Jobs

| Job | Experiments | Nodes | Time (estimate) |
|---|---|---|---|
| Train | D0, D1, D2, D3 | 4 nodes at the same time | ~2 h |
| Eval | D0 to D4 (D4 starts at once) | 5 nodes | ~45 min for each experiment |
| Scaled merge and eval | Best 2 experiments × 3 scales | 2 nodes | ~1 h |

Total: approximately 4 hours after the code is complete. D4 can start before the code is complete.

## Risks

| Risk | Action |
|---|---|
| The tokenizers are different | Stop D2. Run D0, D1, D3 and D4. |
| The top-20 probabilities do not cover all tokens | Normalize over the top 20 tokens (standard for sparse KD). |
| D3 needs 2 forward passes in each step | Train time increases by approximately 30%. Use the same batch. |
| The answers stay short in all experiments | Expected: the data is the same. Answer length is a separate problem (new data). |
| The reservation ends at 18:00 | Without the reservation, the jobs wait in the normal queue. |

## Success criteria

1. Mean v3 win rate (3 seeds) of an experiment without scaling: more than 0.5.
2. Mean v2 win rate: not lower than D0 by more than 0.05.
3. Bad-flag rate on v3: not higher than the base model (0.021).
