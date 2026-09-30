# Iteration 1: gentler LoRA training (2026-09-30)

**TL;DR:** Lowering the learning rate and LoRA rank fixed the endless looping. Win rate against the base model rose from **27.5% to 45%**. That is still below the 50% break-even and the 60% target, so the student is not yet better than its base model.

## What changed vs. iteration 0

| | Iter 0 | Iter 1 |
|---|---|---|
| learning_rate | 2e-4 | **5e-5** |
| lora_r / lora_alpha | 64 / 128 | **16 / 32** (alpha/r ratio kept at 2) |
| Data, eval set, judge | same 40 samples, 10 questions | same |
| Config / run dir | `configs/toy.yaml`, `runs/toy` | `configs/iter1.yaml`, `runs/iter1` |

## Results

| | Iter 0 student | Iter 1 student | Base |
|---|---|---|---|
| Answers hitting the 512-token limit | 9/10 | **1/10** | 0/10 |
| Check pass rate | 0.70 | **1.00** | 0.80 |
| Bad-flag rate | 0.015 | **0.0** | 0.33 |
| Judge win rate vs base | 27.5% | **45.0%** | - |
| `judge_unparsed` | 1 | 0 | - |

Training loss went 1.85 -> 1.63 (iteration 0 reached 1.33), so the model fits the data less tightly, which is what we wanted.

## What we learned
- The looping was caused by over-aggressive training, confirming the iteration 0 hypothesis. It did not need an fp32 eval fix.
- The student now makes no invented flags, but it still loses slightly to the base model on judged quality.
- Base answers are short (median ~80 tokens); the teacher answers the student learned from are long (~890 chars). Next: regenerate with the "short and direct" prompt, and add more data. 40 samples over 6 optimizer steps is very little.
