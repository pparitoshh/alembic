# Iteration 3: stronger training on the 120-sample set (2026-09-30)

**TL;DR:** Same data as iteration 2, but learning rate 1e-4 and 3 epochs. Win rate against the base model jumped to **77.5%** (target: > 60%), from 45%. First run where the student clearly beats its base.

| | Iter 2 | Iter 3 |
|---|---|---|
| lr | 5e-5 | **1e-4** |
| epochs | 2 | **3** |
| r / alpha | 16 / 32 | same |
| Final train loss | 1.60 | 1.39 |
| Answers at 512-token limit | - | 0/10 |
| Student answer length (tokens) | long | 31-95 (base 47-290) |
| Check pass rate | 0.80 | **0.90** |
| Bad-flag rate | 0.052 | 0.056 (base 0.33) |
| Win rate vs base | 42.5% | **77.5%** |
| `judge_unparsed` | 0 | 0 |

Judged both orders: 6 questions won in both/at least one order clearly, 3 ties, 1 loss-heavy question. Config: `configs/iter3.yaml`.

## What we learned
- Iteration 1 fixed looping by training gently; iteration 2 showed gentle + more data still undertrains (16 steps at 5e-5). Iteration 3's stronger settings on the same data were what worked, i.e. the learning rate mattered more than data volume here.
- The student now gives short, direct answers, which the judge prefers over the base's longer ones.
- **Caveats:** only 10 eval questions from one held-out doc, so the score is noisy (+-15 points plausible). The student's script answers are still not correct (e.g. a for-loop instead of `--array`). Next: more eval questions, a repeat with another seed, and checking that it holds on a second held-out doc.
