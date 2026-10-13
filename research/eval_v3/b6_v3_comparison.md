# Eval v3 development comparison

Status: **complete**.

V2 and v3 measure different topic distributions; compare models within each set.
V3 uses source chunks already encountered in historical B6 training. Future training must exclude them.
No model-quality improvement is implied by finishing this evaluation.

| Set | Seed | Questions | Student vs base win rate | Base check pass | Student check pass |
|---|---:|---:|---:|---:|---:|
| v2 | 42 | 66 | 0.3939 | 0.9545 | 0.9394 |
| v2 | 1 | 66 | 0.3864 | 0.9545 | 0.9697 |
| v2 | 2 | 66 | 0.3258 | 0.9545 | 0.9394 |
| v3 | 42 | 200 | 0.4313 | 0.8850 | 0.9150 |
| v3 | 1 | 200 | 0.4338 | 0.8850 | 0.9100 |
| v3 | 2 | 200 | 0.4150 | 0.8850 | 0.9450 |

| V3 topic | Questions per seed | Mean win rate across 3 seeds |
|---|---:|---:|
| accounting | 30 | 0.4111 |
| debug | 20 | 0.3250 |
| dependencies | 30 | 0.3583 |
| job_control | 25 | 0.5000 |
| multi_constraint | 20 | 0.7375 |
| output_mail | 25 | 0.1633 |
| queue_cancel | 20 | 0.4292 |
| resources | 30 | 0.5278 |

These are descriptive scores. Shared sources, repeated seeds, and the two judge orders are correlated; do not treat their verdicts as independent trials or infer a statistically established improvement from a small difference.


## Aggregate interpretation

| Set | Student win rate, mean ± sample SD across three training seeds |
|---|---:|
| v2, 66 questions | 36.87% ± 3.74 pp |
| v3, 200 questions | 42.67% ± 1.02 pp |

The B6 student remains below the 50% pairwise comparison baseline on both sets.
V3 does **not** demonstrate an overall quality improvement. Its deterministic
check pass rate averages 92.33%, compared with 88.50% for the base, but
these checks do not establish semantic correctness or successful Slurm execution.
The three seeds use identical cached base answers. All 600 question/seed pairs
have both judge orders, with zero invalid verdicts (1,200 verdicts total).

The spread above is a sample standard deviation across seeds, not a confidence
interval. Do not pool the 1,200 correlated verdicts as independent trials.
Raw per-question paired verdicts are in `b6_v3_verdicts.jsonl`. Full-answer artifact
hashes, adapter hashes, runtime versions and locations are in `b6_v3_run_manifest.json`.
