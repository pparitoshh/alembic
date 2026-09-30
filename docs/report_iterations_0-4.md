# Report: iterations 0-4, what moved the win rate (2026-09-30)

*Goal (GOAL.md): LoRA-trained Qwen3-0.6B student that beats its own base model, judge win rate above 60% and fewer invented flags. All numbers: student vs. the untrained base, judged by `deepseek-v4.1-flash` in both answer orders (win = 1, tie = 0.5).*

## 1. Bottom line

- The goal is **met on average, but marginally**: over three training seeds the student wins **64.1% (SD 4.8 points)** of 26 judged comparisons. The range is 59.6% to 69.2%, so one seed sits just below 60%.
- The first iteration-3 number (77.5%) was **optimistic**. It came from 10 questions and one seed. On those same 10 questions the three seeds give 65-80% (mean 70.8%). On 16 new questions from a document the student never saw, the mean is only **59.9%**.
- The single change that clearly mattered: **stronger training on the larger dataset** (lr 5e-5 -> 1e-4, epochs 2 -> 3). It lifted the win rate on the same 10 questions from 42.5% to 65-80% for all three seeds. That gap is well outside the seed spread.
- The student also makes fewer invented flags than the base (about 0.065 vs 0.143 of flags used), though the check pass rate is equal (0.92).

## 2. All iterations

| Iter | What changed vs. previous | Data | lr | LoRA r / alpha | Epochs | Optimizer steps | Eval (questions) | Win rate | Looping (answers at 512 tokens) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | first run | 40 | 2e-4 | 64 / 128 | 2 | 6 | 10 | **27.5%** | 9/10 |
| 1 | gentler training | 40 | 5e-5 | 16 / 32 | 2 | 6 | 10 | **45.0%** | 1/10 |
| 2 | 3x data, short-answer teacher prompt | 120 | 5e-5 | 16 / 32 | 2 | 16 | 10 | **42.5%** | not measured |
| 3 | stronger training | 120 | 1e-4 | 16 / 32 | 3 | 24 | 10 | **77.5%** | 0/10 |
| 4 | seed test of iter 3, bigger eval | 120 | 1e-4 | 16 / 32 | 3 | 24 | 26 | **64.1% mean of 3 seeds** | - |

Configs: `configs/toy.yaml` (iter 0), `iter1.yaml`, `iter2.yaml`, `iter3.yaml`, `iter4_s{42,1,2}.yaml`. Per-iteration write-ups: `iter_0_learning.md` to `iter_3_learning.md`.

## 3. Seed-level testing (iteration 4)

Identical data (`iter3` verified.jsonl, 120 samples) and settings; only the training seed differs. Seed 42 is the original iteration-3 adapter, re-judged. Eval set: the original 10 questions (`arr-*`, from `slurm_job_arrays`) plus 16 new ones (`req-*`, from the held-out doc `slurm_requeue_signals`, written before any training on it and excluded from data generation via `eval_docs`).

| Seed | Final train loss | Win rate, all 26 | `arr-*` (10 q) | `req-*` (16 q, new doc) | Student bad-flag rate | `judge_unparsed` |
|---|---|---|---|---|---|---|
| 42 | 1.393 | **69.2%** | 80.0% | 62.5% | 0.069 | 0 |
| 1 | 1.396 | **63.5%** | 67.5% | 60.9% | 0.061 | 1 |
| 2 | 1.381 | **59.6%** | 65.0% | 56.2% | 0.069 | 0 |
| **Mean (SD)** | 1.39 | **64.1% (4.8)** | 70.8% (8.0) | 59.9% (3.3) | 0.066 | - |

Base for comparison: check pass rate 0.92, bad-flag rate 0.143. Student check pass rate: 0.92 for all seeds. Student answers are short: median about 150 characters vs. 344 for the base.

What this shows:
- **Seed noise is about +-5 points** (SD) on 26 questions, and 8 points on the original 10. A gap between two settings smaller than roughly 10 points cannot be trusted with this eval.
- **Ranking is consistent across seeds:** all seeds beat the base overall, and all do better on `arr-*` than on `req-*`. The new-document score (59.9%) is the more honest estimate of generalisation, since the `arr-*` questions were the ones we looked at while tuning.
- Seed 42 (the one we tuned on) was the best seed, a mild case of picking the winner.
- Judge nondeterminism is small but real: the same seed-42 adapter scored 77.5% (iteration 3, judged once) and 80.0% on the same 10 questions here. One judge reply in seed 1 was unparsed and counted as a tie.

## 4. Which parameters moved the lever the right way

Honest caveat first: **each step changed several things at once, and no single parameter was ablated.** What follows separates what the data supports from what is inference.

| Change | Effect on win rate | Evidence quality |
|---|---|---|
| **lr 2e-4 -> 5e-5, r 64 -> 16, alpha 128 -> 32** (iter 0 -> 1) | 27.5% -> 45.0%; looping 9/10 -> 1/10 | Clear on looping (9/10 vs 1/10). Three params moved together, so it is not known which one mattered. |
| **3x data + "short and direct" teacher prompt** (iter 1 -> 2) | 45.0% -> 42.5% | **No detectable effect.** Within noise. Data volume alone did not help; it also tripled the optimizer steps to 16. |
| **lr 5e-5 -> 1e-4 and epochs 2 -> 3** (iter 2 -> 3) | 42.5% -> 65-80% (three seeds, same 10 questions) | **Strongest result.** All three seeds beat iteration 2 by 22+ points, larger than the seed SD. lr and epochs moved together, so not separable. |
| Lower rank (r16 vs r64) | unknown | Not isolated; alpha/r ratio was held at 2 throughout. |

Interpretation (a hypothesis, not tested by ablation): the student needs **enough total training signal without the aggressive update size that causes looping.** Iterations 2 and 3 have the same rank and data; only lr x steps changed (5e-5 x 16 steps -> 1e-4 x 24 steps) and the win rate jumped. Iteration 0 had a large lr (2e-4) at high rank with only 6 steps and looped. So the useful region looks like: moderate lr (about 1e-4), low rank (16), and 20+ steps. Whether 2e-4 at rank 16 or 4-5 epochs helps further is untested.

Directions that appear to help the win rate:
- More optimizer steps at a moderate lr (6 -> 16 -> 24 steps with lr 5e-5 -> 1e-4).
- Low LoRA rank with alpha = 2r.
- Short, direct teacher answers: the student now answers in about 150 characters vs. the base's 344. Whether the judge rewards this brevity independent of correctness is not known.

Directions that do not help (or hurt):
- High lr and rank with few steps (looping).
- More data at the same low training intensity.

## 5. Remaining weaknesses

- Script answers are often still wrong (e.g. a shell for-loop instead of `#SBATCH --array`), and the student sometimes invents options.
- The new-document score (59.9%) is right at the 60% target line.
- Eval is small: 26 questions from two hand-written documents; the judge is one model; both answer orders are judged but there is no second judge.
- The flag list (`data/slurm_flags.txt`) is hand-written, so `check_pass_rate` and `bad_flag_rate` can mislabel correct flags.

## 6. Infrastructure changes made during iteration 4

- `train.seed` config option (default 42, the previous behaviour).
- **API retries with tenacity:** the judge endpoint returned `429: Endpoint is unavailable` during the seed runs and crashed two evaluations (those seeds were re-run). `Teacher` now retries rate-limit, 5xx, timeout and connection errors with exponential backoff and jitter (up to 8 attempts, capped at 60 s), instead of the OpenAI client's default two retries. Added `tenacity>=9.0` as a dependency. Tested with a fake endpoint that fails twice, then succeeds.
- New held-out document `data/seeds/slurm_requeue_signals.md` and eval sets `data/eval/eval_req.jsonl` (16 q) and `eval_all.jsonl` (26 q, combined).
- Raw seed summaries: `docs/results/iter4_seed_summaries.json`.
- A process slip worth recording: running plain `uv sync` removed the `train` extra (torch etc.) from the shared venv, which silently failed the first seed runs. Use `uv sync --extra train`.

## 7. Recommended next steps (in order)

1. **Ablate lr and epochs separately** (lr 1e-4 with 2 epochs; lr 5e-5 with 4-5 epochs), 3 seeds each on `eval_all`. This is the one open question about what actually caused the jump.
2. **Grow the eval set** to 50+ questions across at least 3 held-out documents so a 5-point difference becomes detectable.
3. **Target script/`--array` errors** with more `script` and `howto` training examples and stricter verification of teacher scripts.
4. **Rebuild `slurm_flags.txt` from the real man pages.**
5. Only then scale data toward ~5k samples.
