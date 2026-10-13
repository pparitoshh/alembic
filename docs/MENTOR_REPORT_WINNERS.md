# DistillKit: Models That Win Against the Base Model

Date: 13 October 2026, 16:00. Team: Alembic. Mentor: Simeon Harrison.

This is a short version of [MENTOR_REPORT_v2.md](MENTOR_REPORT_v2.md). It shows only the student models
that win against the base model, and the teacher model for comparison.

## Terms

| Term | Meaning |
|---|---|
| Base model | Qwen3-4B-Instruct-2507, without changes. Win rate 0.50 by definition. |
| Teacher model | Qwen3-32B-AWQ. It writes the training examples. |
| Win rate | The fraction of judge comparisons (gpt-oss-20b, both answer orders) that a model wins against the base model. A tie counts as 0.5. |
| v2 test set | 66 questions on topics that we keep out of the training data (job arrays, requeue, signals). Margin of error ±0.12. |
| v3 test set | 200 questions on topics that are in the training data. The source text of these questions is not in the training data. |
| Coverage | The fraction of the key terms of the reference answer (commands, options, variable names) that the answer contains. |

## How to read a win rate

The judge reads the question, the reference answer and 2 answers to the same question:

- one answer from the **tested model**: a row of the results table (x0.1, x0.2, x0.3, D3 or the teacher model);
- one answer from the **base model** (Qwen3-4B-Instruct-2507, without changes).

Each tested model has its own win rate. For example, the teacher row (0.585) compares teacher answers with base
answers. The D3 row (0.525) compares D3 answers with the same base answers.

The reference answer is a correct sample answer that is stored with each test question (field `reference` in
`data/eval/*.jsonl`). The judge uses it as the correct facts. It does not compare the 2 answers with the reference
answer for a score. It only decides which of the 2 answers agrees better with the reference answer.

| Test set | Source of the reference answers |
|---|---|
| v3 | The teacher model wrote them from the Slurm 26.05.4 documents. Gemma-4-26B checked each answer against its source text, and automatic checks tested the flags. Each reference names its source part. No person checked them. |
| v2 | Short answers with the key facts of the question. |

Example (v3-001): "Start a job only after jobs 123 and 456 complete successfully." The reference answer is
`sbatch --dependency=afterok:123:456 my_job_script.sh`. An answer that gives `afterok` with these 2 job IDs agrees
with the reference answer. An answer that gives `afterany` does not agree.

Models wrote the v3 reference answers. If a reference answer has an error, the judge can follow that error.

The judge gives each question 2 times, with the answers in the 2 orders. This removes the effect of the position.
In the tables below, "model" means the tested model.

Each judge verdict gives points to the model:

| Verdict | Points |
|---|---|
| The model answer is better | 1 |
| Tie | 0.5 |
| The base model answer is better | 0 |

The question score is the mean of its 2 verdicts. The win rate is the mean of all question scores.

Example with 4 dummy questions:

| Question | Verdict, order 1 | Verdict, order 2 | Question score |
|---|---|---|---|
| Q1 | Model better (1) | Model better (1) | 1.0 |
| Q2 | Base better (0) | Base better (0) | 0.0 |
| Q3 | Model better (1) | Base better (0) | 0.5 |
| Q4 | Model better (1) | Tie (0.5) | 0.75 |
| **Win rate** | | | (1.0 + 0.0 + 0.5 + 0.75) / 4 = **0.5625** |

How to read the values:

- 0.50: the model is equal to the base model. This is the reference point.
- More than 0.50: the model is better. Less than 0.50: the base model is better.
- Teacher model on v3, 0.585: the teacher model gets 117 of 200 points (0.585 × 200). The base model gets the other 83 points.
  If no question has a tie, the teacher model wins 117 questions and loses 83. That is 34 more wins than losses.
- x0.2 on v3, 0.528: 105.6 of 200 points. If no question has a tie, x0.2 wins approximately 11 more questions than it loses.

## Results

All values are means of 3 seeds (42, 1 and 2). A model is in this table if its v3 win rate is more than 0.5.

| Model | Method | v2 win rate | v3 win rate | v3 median length (characters) | v3 coverage |
|---|---|---|---|---|---|
| Base model | No training | 0.50 | 0.50 | 1,619 | 0.61 |
| **Teacher model** | Qwen3-32B-AWQ (the source of the training data) | **0.689** | **0.585** | – | – |
| **x0.2** | B7 adapter at 20%: W = base + 0.2 × (B7 − base). No new training | 0.481 | **0.528** | 1,220 | 0.61 |
| **D3** | Training with a KL penalty to the base model: CE + 0.1 × KL(student ‖ base), 12,892 good-quality examples | 0.442 | **0.525** | 973 | 0.62 |
| x0.1 | B7 adapter at 10% | 0.481 | 0.522 | 1,526 | 0.60 |
| x0.3 | B7 adapter at 30% | 0.443 | 0.510 | 886 | 0.60 |

For comparison, full training (B7) has v2 0.398, v3 0.438 and v3 coverage 0.53.

## Findings

1. No student model wins on v2. On these topics, the training data has no facts, so the best possible result is near 0.5.
2. x0.2 and D3 win on v3 by approximately 0.03. This win is small.
3. Both winners keep the key terms of the base model (coverage 0.61 to 0.62). Full training (B7) loses them (0.53).
4. The win does not come from longer answers. In each model, the answer length does not predict the judge score
   (Spearman rho -0.14 to 0.08). The coverage predicts it (rho 0.14 to 0.44).
5. The teacher model wins on both test sets. Thus, distillation can give a better model. Our student models get
   only a small part of the teacher gain.
6. D3 is the only trained model that wins. x0.2 wins without new training.

## Methods that did not win

| Method | v2 | v3 |
|---|---|---|
| Logit distillation from the teacher top-20 tokens (D2) | 0.361 | 0.450 |
| DPO on x0.2 (D5) | 0.442 | 0.421 |
| Early stop of B7 at steps 300 to 700 (D4) | 0.374 to 0.396 | 0.406 to 0.409 |
| Low learning rate, rank 8, half an epoch (B8a) | – | 0.347 |

## Next steps

1. Scaled merges of D3 (20%, 30% and 50%). x0.2 shows that scaling keeps the base facts. A scaled D3 can do better.
2. Training answers with full explanations (the teacher prompt now asks for concise answers).
3. RAG on the Slurm documents for the v2 topics. Training cannot add facts that are not in the data.

## Questions for the mentor

1. Is a scaled model (x0.2) an acceptable final model, or must the improvement come from the training (D3)?
2. A v3 win of approximately 0.03 is small. Do you accept it as a result, or must we make the v3 test set larger?
