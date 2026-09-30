# Iteration 2: 3x more data, same gentle settings (2026-09-30)

**TL;DR:** 120 short-answer samples (was 40) with iteration 1's settings gave a **42.5%** win rate vs 45% before. Not a real change: with 10 eval questions, one question moves the score 5 points. Data alone did not help; the model was likely undertrained.

| | Iter 1 | Iter 2 |
|---|---|---|
| Samples | 40 | 120 (60 questions x 2; all passed verify) |
| Teacher prompt | old (long answers) | "short and direct" |
| Optimizer steps | 6 | 16 |
| lr / r / alpha / epochs | 5e-5 / 16 / 32 / 2 | same |
| Win rate | 45.0% | 42.5% |
| Check pass rate | 1.00 | 0.80 |
| Bad-flag rate | 0.0 | 0.052 |

Generation took 12 min at concurrency 12. Judge logs show the student still invents `--array-*` options on script questions and is often longer than the base. Not committed to `dev` as a success. Config: `configs/iter2.yaml`.
