# Controlled QDoRA comparison: 12,092 examples

Status: submitted, no comparison results yet. Six separate training runs use the same immutable certified snapshot and training implementation. Explicit random seeding happens before model/adapter initialization.

| Learning rate | Seeds | Other settings |
|---|---|---|
| 0.0001 | 42, 43, 44 | Qwen3-4B-Instruct-2507, NF4 QDoRA, rank 16, alpha 32, 3 epochs, effective batch 16, BF16, max length 2,048 |
| 0.00005 | 42, 43, 44 | Identical |

The already queued 12,092-example seed-42 run is included; five additional runs were submitted. `manifest.json` records every training/evaluation job ID, frozen input hashes, and the separate aggregation job. The 11,167-example published pilot is evaluated separately and excluded from these matched statistics.

## Evaluation

Each successful training job triggers base/student answer generation on the same 106 normal and 23 first-response tool development tasks. GPT-OSS-20B and Gemma 4 independently score the same cached normal answers in both orders. Each judge must first reach at least 80% calibration accuracy and 80% position consistency, with zero unparsed verdicts, on the separate 24-pair calibration set. These thresholds were fixed before this experiment's results. Failed calibration blocks headline comparison statistics; no threshold is relaxed automatically.

The report requires all three seeds per learning rate and reports mean and sample standard deviation, plus paired per-seed differences between learning rates. It rejects changed training implementations or hyperparameters beyond seed and learning rate. The 22-item final source-family holdout is not selected for this comparison and must not guide tuning.

Validation: 91 production evaluation/scorer tests passed on Leonardo; configuration/input-hash preflight passed. Exact evaluation-question, known protected-source-family and held-out document-ID overlap checks found zero matches in the training snapshot. The aggregator was checked for missing runs, matched seed differences and calibration-failure gating.

## Limits

These are development measurements, not a blind final benchmark or evidence of improvement until results exist. Three-seed spread is not a significance claim. Static flag/Bash checks and isolated tool mocks do not establish compiled CUDA/MPI correctness, actual Slurm execution or multi-turn task success. Exact overlap checks do not guarantee freedom from semantic contamination. Generated answers are capped at 512 new tokens; the production path does not retain finish reasons.

## Execution record

The copied scripts record the actual Leonardo experiment under `student-experiments-v1`; they use the frozen repository snapshot in `manifest.json` for evaluation and the existing frozen campaign training implementation. The training wrapper is deployed as `dependency-pipeline-v1/train12092.py`. These are environment-specific execution records, not a new portable package entry point. Live jobs may wait for the account's concurrent-node allocation; submitted does not mean running or completed.

A local completion monitor checks every five minutes after the aggregation job. It publishes the report (including incomplete or calibration-blocked outcomes), evaluation summaries and checksum-verified completed adapters. This requires the publication Mac to remain awake, online and authenticated. Full generated evaluation answers remain on Leonardo; published summaries do not expose final-test content.
