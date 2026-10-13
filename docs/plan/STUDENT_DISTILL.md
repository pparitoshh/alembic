# Plan: student distillation experiments on the existing B7 data

Status: plan, 2026-10-13. Nothing in this plan is submitted. Branch `feat/data-hpc10k`.
Results go to `docs/TRAINING_PROGRESS.md` and `docs/MENTOR_REPORT_v2.md` (means of 3 seeds only).

## Goal

The student model must win against the base model on the v3 test set (win rate more than 0.5, mean of
3 seeds). It must not become worse on the v2 test set by more than 0.05.

## Scope

- We change only the training method. We do not generate new training data.
- All experiments use only the good-quality subset of the B7 data (section "Training data").
  The source is `runs/b7_hpc10k/verified.jsonl` (20,472 examples, decontam complete).
- The teacher top-20 token probabilities for this data are on disk:
  `runs/hpc10k_*/teacher_logprobs.jsonl.gz` and `runs/gen_slurm_tools*/teacher_logprobs.jsonl.gz`.
- DPO is experiment D5 in this plan (details in the section "D5: DPO"). D5 uses the teacher answers
  that we have. The student writes the rejected answers. The teacher writes no new answers.

## Facts that the plan uses

| Fact | Value | Source |
|---|---|---|
| B7 (SFT, 4-bit, full change) | v2 0.398, v3 0.438 | 3 seeds |
| B7 with 20% of the change | v2 0.481, v3 0.528 | 3 seeds |
| B8a (lr 2e-5, rank 8, half epoch, 4-bit) | v3 0.347 | 3 seeds |
| Teacher model | v2 0.689, v3 0.585 | – |
| Median answer length: base / training data / B7 | 1,381 / 442 / 313 characters | v2 |
| Validation loss minimum | approximately half of the epoch | B6 |

Conclusion: the full SFT change removes base knowledge. A small change wins. A gentle run (B8a) does not
win. Thus, the problem is the direction of the change, not only its size.

## Training data

All training experiments (D0, D2, D3, D5) use the same good-quality subset. Thus, a difference in a
result comes from the method, not from the data.

A row is good quality when all of these are true:

1. The row is in `runs/b7_hpc10k/verified.jsonl` (verified and decontam complete).
2. The row has 1 assistant answer and no tool call.
3. The row has teacher top-20 probabilities.
4. The teacher token ids agree exactly with the student tokenization of the answer.

Result of the first check on 13 October (`$WORK/alembic/jobs/kd_check.py`, first file for each id):

| Item | Rows |
|---|---|
| B7 rows | 20,472 |
| Prose rows | 19,772 |
| Prose rows with teacher probabilities | 12,997 |
| Good quality (exact token agreement, first file only) | 10,867 |
| Removed: no teacher probabilities | 6,775 |
| Removed: tokens do not agree | 2,130 |
| Removed: tool and multi-turn rows | 700 |

The tokenizers are the same (151,669 tokens, the same special tokens). 4,112 row ids occur in more than
1 run directory. The check used the first file for each id. A match on the answer tokens can recover
some of the 2,130 rows. Do this check before the subset is final.

Final subset (`$WORK/alembic/jobs/build_good.py`, 13 October 12:38): the script tries every file for a
duplicate id and keeps the entry whose tokens agree.

| Item | Rows |
|---|---|
| **Good quality: `runs/b7_good/verified.jsonl`** | **12,892** |
| Removed: no teacher probabilities | 6,775 |
| Removed: tool and multi-turn rows | 700 |
| Removed: tokens do not agree | 105 |

`runs/b7_good/teacher_logprobs.jsonl.gz` has 1 entry for each good row. `runs/b7_good/removed.jsonl` has the
removed ids and the reason for each.

## Experiments

All experiments train from the base model with bf16 weights (`configs/qwen3_4b_dora_bf16.yaml` as the
start). Each experiment uses 1 node: 3 seeds (42, 1, 2), 1 GPU for each seed (`fan_out_seeds`).
The experiments do not depend on each other. Thus, all can run at the same time.

| # | Name | Change from B7 | Question | Code | Train time |
|---|---|---|---|---|---|
| D0 | B9 control | bf16 weights, not 4-bit, good-quality subset | Is bf16 SFT different from 4-bit SFT? Reference for D2 to D5. | Config only | ~1.5 h |
| D2 | Logit KD | Loss = 0.5 × CE + 0.5 × KL(teacher top-20 ‖ student), temperature 1 | Do the teacher probabilities give more information than the text? | New loss in `train.py` | ~1.5 h |
| D3 | KL to base | Loss = CE + λ × KL(student ‖ base), λ = 0.1; base = the model with the adapter off | Does a penalty during training keep the base knowledge (the same effect as scaling)? | New loss in `train.py` | ~2 h |
| D4 | Early checkpoints | No training: evaluate the saved B7 adapters at steps 300, 500 and 700 | Is a shorter B7 run better than the full run? | None (eval only) | 0 |
| D5 | DPO | Start model: B7 x0.2. Preferred: the verified teacher answer. Rejected: a student answer with an invented flag or with missing flags. New LoRA, beta 0.1, lr 5e-6 | Does DPO remove invented flags and keep the x0.2 win? | `dpo.py`, `merge.py` | ~15 min mining + ~30 min |

After the evaluation, apply the scaled merge (`base + s × (model − base)`, s = 0.2, 0.3, 0.5) to the best 2 of
D0, D2 and D3. This costs only answer and judge time.

## D5: DPO

| Item | Value |
|---|---|
| Start model and reference model | B7 x0.2 for each seed: `base + 0.2 × (B7 − base)`, saved in bf16 |
| Questions | Questions from the good-quality subset `runs/b7_good/verified.jsonl`, 6,000 to 12,892 for each seed |
| Preferred answer | The verified teacher answer. Skip the question if this answer has an invented flag. |
| Rejected answer | A start-model answer (temperature 0.8, 2 samples) with an invented flag, or with less than half of the teacher answer flags (minimum 2 flags) |
| Target | At least 800 pairs for each seed. The first mining run (B7 x1.0) found 858 pairs from 6,000 questions. |
| Trainer | TRL `DPOTrainer`, sigmoid loss, beta 0.1 |
| Adapter | New LoRA, rank 16, alpha 32, all linear layers. The reference is the start model with the adapter off. |
| Training | lr 5e-6, cosine, 3% warmup, 1 epoch, effective batch 16, maximum length 2,048 |
| Output | `runs/d5_dpo/seed_<n>/pairs.jsonl` and the adapter |
| Commands | `python -m distillkit.merge BASE ADAPTER OUT --scale 0.2`, `python -m distillkit.dpo mine ...`, `python -m distillkit.dpo train ...` |

The D5 settings are command arguments, not a `dpo` section in `Config`. Thus, `config.py` and its
binding in `data/eval/dataset_v1/manifest.json` do not change again.

Compare 3 models for each seed: base model, B7 x0.2 and B7 x0.2 + DPO.

## Length analysis

Measured on 13 October 13:10 (`runs/scaled_eval/*_answers.json`). Median answer length in characters:

| Model | v2 | v3 |
|---|---|---|
| Base model | 1,381 | – |
| x0.1 (seed 42 / seed 1) | 1,287 / 1,226 | 1,542 / 1,526 |
| x0.2 (seed 1) | 1,048 | 1,272 |
| x0.3 (seed 1) | 632 | 884 |
| x0.4 (seed 1) | 530 | 786 |
| B7 (x1.0) | 313 | – |
| D4: B7 step 300 (seed 42 / 1 / 2) | 281 / 330 / 314 | 354 / 390 / 379 |

1. The answers become longer when the scale becomes smaller. At x0.1 the length is almost the base length.
2. The win rate also increases when the scale becomes smaller. Thus, a part of the x0.2 win can come from
   the longer answers, not from better facts.
3. The B7 step-300 answers (27% of the run) are as short as the full B7 answers. The model learns the short
   style early. Thus, a shorter run does not make the answers longer; the weight scaling does.
4. On B7, the length had no constant relation to the result (mentor report, section 4.2). We did not test
   this on the scaled models.

Analysis to do when the judge results of x0.1 and D4 are complete:

1. For each model and question, record the student length, the base length and the judge score.
2. Calculate the win rate in length bins (student length / base length: < 0.5, 0.5–1, > 1).
3. Calculate the correlation between the length ratio and the score.
4. Calculate the key-term coverage (fraction of the reference answer key terms), as in section 4.2 of the
   mentor report.
5. Record the median length for all experiments (D0 to D5) in the results table.

Decision rule:

| Result | Conclusion | Action |
|---|---|---|
| The score follows the length, and the coverage does not increase | The win comes from the length | Short training answers are the main problem. D2, D3 and D5 use the same answers, so they cannot fix it. Plan new data with full answers. |
| The coverage increases with the score | The win comes from more facts | Continue with D2, D3 and D5. |

## Phase 2 (only if D2, D3 or D5 gives a result above 0.5)

| # | Name | Change | Note |
|---|---|---|---|
| D6 | D2 + D3 | Logit KD and KL to base in one loss | Combine the 2 best changes |
| D7 | DPO on the best of D0, D2 and D3 | Mine the pairs again from that model | DPO works best on rejected answers from the model that it trains |
| D8 | On-policy distillation (GKD) | The student writes the answers; the teacher scores each token | The teacher must run during training (2 GPUs for each seed) |

## Run tracker

Update this table at each job change. Paths are relative to `$WORK/alembic/alembic` on Leonardo.
Ad-hoc job scripts are in `$WORK/alembic/jobs/` (not in git). All jobs use the `s_tra_ncc` reservation on 13 October.

| Experiment | Job | Job id | Submitted | Status | Output |
|---|---|---|---|---|---|
| Scale x0.1, seed 42 | `scaled_answers.sbatch` (SCALES=0.1) | 60294836 | 13 Oct 12:38 | answers complete | `runs/scaled_eval/b7s42_0.1_answers.json` |
| Scale x0.1, seed 1 | `scaled_answers.sbatch` (SCALES=0.1) | 60294842 | 13 Oct 12:38 | answers complete | `runs/scaled_eval/b7s1_0.1_answers.json` |
| Scale x0.1, seed 2 | `scaled_answers.sbatch` (SCALES=0.1) | 60294850 | 13 Oct 12:38 | answers complete | `runs/scaled_eval/b7s2_0.1_answers.json` |
| D4, seed 42 (steps 300, 500, 700) | `ckpt_answers.sbatch` | 60294853 | 13 Oct 12:38 | running | `runs/scaled_eval/b7s42ck<N>_1.0_answers.json` |
| D4, seed 1 (steps 300, 500, 700) | `ckpt_answers.sbatch` | 60294857 | 13 Oct 12:38 | running | `runs/scaled_eval/b7s1ck<N>_1.0_answers.json` |
| D4, seed 2 (steps 300, 500, 700) | `ckpt_answers.sbatch` | 60294859 | 13 Oct 12:38 | running | `runs/scaled_eval/b7s2ck<N>_1.0_answers.json` |
| Judge for x0.1 and D4 | `judge_dirs.sbatch` (afterok on the 6 jobs above) | 60294860 | 13 Oct 12:38 | pending (dependency) | `$WORK/alembic/wt-eval-v3/runs/judge_b7s<n>_t0.1_<v2,v3>/`, `judge_b7s<n>ck<N>_t1.0_<v2,v3>/` |
| D2, D3 GPU smoke test (~23 steps each, 1 GPU each) | `$WORK/alembic/jobs/smoke_kd.sbatch` | 60300003 | 13 Oct 13:03 | failed: D3 `logits=None` (TRL `chunked_nll`); D2 ran with no KD (assistant mask has `<|im_end|>` + newline). Fixed in a9b7ce7 | `runs/smoke_d2`, `runs/smoke_d3` (in the worktrees), logs `slurm/logs/smoke-*-60300003.log` |
| D2, D3 GPU smoke test, try 2 | `$WORK/alembic/jobs/smoke_kd.sbatch` | 60301833 | 13 Oct 13:14 | passed: D2 `kd_loss` 0.71–0.82, D3 `base_kl` 0.42–0.47 | try 1 outputs moved to `runs/smoke_*_try1_60300003` |
| D0 train (3 seeds) | `slurm/train.sbatch` in `wt-d0-bf16` | 60302556 | 13 Oct 13:20 | running | `runs/d0_bf16/seed_<n>/` |
| D2 train (3 seeds) | `slurm/train.sbatch` in `wt-d2-kd` | 60302558 | 13 Oct 13:20 | running | `runs/d2_kd/seed_<n>/` |
| D3 train (3 seeds) | `slurm/train.sbatch` in `wt-d3-basekl` | 60302559 | 13 Oct 13:20 | running | `runs/d3_basekl/seed_<n>/` |
| D5 prepare (merge x0.2, mine pairs) | `slurm/d5_prepare.sbatch` in `wt-d5-dpo` | 60302560 | 13 Oct 13:20 | running | `runs/d5_dpo/seed_<n>/start`, `pairs.jsonl` |
| D5 train | `slurm/d5_train.sbatch` (afterok 60302560) | 60302579 | 13 Oct 13:20 | pending (dependency) | `runs/d5_dpo/seed_<n>/adapter` |
| D5 eval (answers) | `slurm/d5_eval.sbatch` (afterok 60302579) | 60302582 | 13 Oct 13:20 | pending (dependency) | `runs/scaled_eval/b7s<n>d5_answers.json` |
| D5 judge | `judge_dirs.sbatch` (afterok 60302582) | 60302589 | 13 Oct 13:20 | pending (dependency) | `judge_b7s<n>d5_t1.0_<v2,v3>/` |

Submission (approved 13 October; submitted 13:20 after smoke try 2 passed). Rule: both smoke runs exit 0, and `kd_loss` and `base_kl` are finite. Leonardo worktrees: `$WORK/alembic/wt-<branch>`,
each with `runs` linked to the shared `$WORK/alembic/alembic/runs`.

```bash
source $WORK/alembic/me.env; export SBATCH_RESERVATION=s_tra_ncc   # on-site days only
# D0, D2, D3: training data = the good-quality subset (a link, no copy)
for x in d0-bf16:d0_bf16:qwen3_4b_d0_bf16 d2-kd:d2_kd:qwen3_4b_d2_kd d3-basekl:d3_basekl:qwen3_4b_d3_basekl; do
  IFS=: read wt rd cf <<< "$x"; W=$WORK/alembic/wt-$wt
  mkdir -p $W/runs/$rd && ln -sn ../b7_good/verified.jsonl $W/runs/$rd/verified.jsonl
  (cd $W && REPO=$W PYTHONPATH=$W/src CONFIG=configs/$cf.yaml RUN_DIR=runs/$rd sbatch slurm/train.sbatch)
done
# D5: prepare -> train -> eval, then the judge
W=$WORK/alembic/wt-d5-dpo; cd $W
a=$(REPO=$W sbatch --parsable slurm/d5_prepare.sbatch)
b=$(REPO=$W sbatch --parsable --dependency=afterok:$a slurm/d5_train.sbatch)
c=$(REPO=$W sbatch --parsable --dependency=afterok:$b slurm/d5_eval.sbatch)
cd $WORK/alembic/wt-eval-v3 && sbatch --dependency=afterok:$c $WORK/alembic/jobs/judge_dirs.sbatch
```

To do: the evaluation of D0, D2 and D3 is a separate job after the training (`slurm/evaluate.sbatch` with the same
`REPO`, `CONFIG` and `RUN_DIR`).

Monitor: `squeue -u $USER`; the result is `student_vs_base_win_rate` in each `eval_summary.json`.
No job overwrites existing adapters, training data or evaluation data: each job writes only new files,
and the judge skips directories that already have a valid summary.

Note: D4 and the scale choice use the v3 results to select a model. Report them as development results.

## Branches and worktrees

`feat/data-hpc10k` is the supervisor branch. It holds this plan, the shared code and the results.
Each experiment has its own branch from the supervisor branch and its own worktree.

| Experiment | Branch | Local worktree |
|---|---|---|
| D0 | `exp/d0-bf16` | `../alembic-wt/d0-bf16` |
| D2 | `exp/d2-kd` | `../alembic-wt/d2-kd` |
| D3 | `exp/d3-basekl` | `../alembic-wt/d3-basekl` |
| D5 | `exp/d5-dpo` | `../alembic-wt/d5-dpo` |
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
| `configs/qwen3_4b_d2_kd.yaml`, `_d3_basekl.yaml` | One config for each experiment (D0 uses `qwen3_4b_dora_bf16.yaml`) |
| `src/distillkit/dpo.py` | D5: pair mining (from `$WORK/alembic/jobs/dpo_mine.py`, flag names without `--`) and `DPOTrainer` training |
| `src/distillkit/merge.py` | Scaled merge `base + s × (model − base)` in bf16 (from `scaled_answers.py`) |
| `tests/test_kd.py` | Hermetic tests: token alignment, KL value on a small known case, loss = CE when the weights are 0 |

## Checks before the first job

1. Done: the tokenizers are the same.
2. Done: 12,997 of 20,472 rows have teacher probabilities (0.635).
3. Done: 12,892 rows agree exactly after the match of duplicate ids. The subset is final (`runs/b7_good`).
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
| Train | D0, D2, D3, D5 (mining first) | 4 nodes at the same time | ~2 h |
| Eval | D0, D2 to D5 (D4 starts at once) | 5 nodes | ~45 min for each experiment |
| Scaled merge and eval | Best 2 experiments × 3 scales | 2 nodes | ~1 h |

Total: approximately 4 hours after the code is complete. D4 can start before the code is complete.

## Risks

| Risk | Action |
|---|---|
| DPO makes the answers shorter (preferred answers: median 442 characters) | Measure the median length. Add a rule: reject a student answer with less than half of the teacher key terms. |
| The subset has no tool rows | The tool slice of the evaluation can become worse. Compare it with B7. |
| The subset (12,892 rows) is 63% of B7 | Less data can lower the v3 result. D0 shows this effect. |
| The top-20 probabilities do not cover all tokens | Normalize over the top 20 tokens (standard for sparse KD). |
| D3 needs 2 forward passes in each step | Train time increases by approximately 30%. Use the same batch. |
| The answers stay short in all experiments | Expected: the data is the same. Answer length is a separate problem (new data). |
| The reservation ends at 18:00 | Without the reservation, the jobs wait in the normal queue. |

## Success criteria

1. Mean v3 win rate (3 seeds) of an experiment without scaling: more than 0.5.
2. Mean v2 win rate: not lower than D0 by more than 0.05.
3. Bad-flag rate on v3: not higher than the base model (0.021).
