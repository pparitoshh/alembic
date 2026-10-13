# DistillKit: Status Report for the Mentor (Version 2)

Date: 13 October 2026, 12:10. Team: Alembic. Mentor: Simeon Harrison.

This version of the report uses ASD-STE100 Simplified Technical English.

## Terms

| Term | Meaning in this report |
|---|---|
| Base model | Qwen3-4B-Instruct-2507, without changes. |
| Student model | The base model after our fine-tuning. |
| Teacher model | Qwen3-32B-AWQ. It writes the training examples. |
| Judge | gpt-oss-20b. It compares two answers to the same question. |
| Win rate | The fraction of comparisons that the student model wins. A tie counts as 0.5. 0.5 is equal to the base model. |
| Round | One training run with 3 seeds (42, 1 and 2). Rounds have the names B1 to B8a. |
| v2 test set | 66 questions on 2 topics that we keep out of the training data: job arrays and requeue/signals. |
| v3 test set | 200 questions on 8 topics that are in the training data. The source text of these questions is not in the training data. |
| Bad-flag rate | The fraction of `sbatch` flags in the answers that do not exist in Slurm. |

## 1. Current status

The student model does not win against the base model. Our target is a win rate of more than 0.5.

- On the v2 test set, B7 has a win rate of 0.398.
- On the v3 test set, B7 has a win rate of 0.450.
- Section 3 shows all results in one table. All values are means of the seeds.
- From B1 to B7, the v2 win rate increased from 0.324 to 0.398.
- The increase from B6 (0.369) to B7 (0.398) is less than the margin of error (±0.12).

We found two causes:

1. The v2 test set measures topics that we do not train. The student model cannot learn these topics from our data.
2. The student model also loses on the v3 test set, which has topics that we train. Thus, the training method also has a problem.

We do not have these results yet:

- The complete B7 result on the v3 test set (1 of 3 seeds runs again).
- The result of the teacher model against the base model. If the teacher model does not win, the student model cannot win by a copy of the teacher model.

## 2. How the system operates

```
 Slurm/HPC documents --> Teacher writes  --> Reviewer  --> Checks and   --> Student     --> Judge compares
 (34 documents; the      questions and       checks the    filters        model trains    student model and
 v2 topics are removed)  answers             answers                                      base model
```

| Step | Model | Hardware | Time |
|---|---|---|---|
| Generate, review and check the data | Teacher model; reviewer Gemma-4-26B | 1 A100 GPU for each part of the data | 5 to 6 hours for each part |
| Train | Base model with QDoRA (4-bit weights, rank 16) | 1 A100 GPU for each seed | 2 hours 23 minutes to 2 hours 46 minutes (B7, 20,472 examples) |
| Answer and judge | Base model, student model and judge | 1 A100 GPU for each seed | Approximately 20 minutes for 66 questions |

Training settings for B2 to B7:

- Learning rate: 5e-5.
- Epochs: 1.
- Effective batch size: 16.
- Maximum length: 2,048 tokens.
- The loss uses only the assistant answers.

How the judge scores:

- The judge compares each pair of answers 2 times. Each answer is first 1 time.
- A win is 1. A tie is 0.5. A loss is 0.
- The judge reads a reference answer. The judge must give a low score to invented options. The judge must not give a high score to a long answer.

## 3. What we did

### 3.1 All results

All values are means of the seeds. The base model has a win rate of 0.50 because it is the comparison point.

| Date | Round | Training data | Change | v2 win rate | v3 win rate | v2 check pass | v3 check pass | v2 bad-flag rate | v3 bad-flag rate |
|---|---|---|---|---|---|---|---|---|---|
| – | Base model | – | – | 0.50 | 0.50 | 0.955 | 0.885 | 0.025 | 0.021 |
| 10 Oct | B1 | 4,775 general HPC examples | Learning rate 1e-4, 3 epochs | 0.324 | – | 0.864 | – | 0.289 | – |
| 10 Oct | B2 | Same as B1 | Learning rate 5e-5, 1 epoch | 0.326 | – | 0.899 | – | 0.220 | – |
| 10 Oct | B3b | B2 data and 182 Slurm examples (3 copies each) | Same as B2 | 0.359 | – | 0.924 | – | 0.059 | – |
| 10 Oct | B3b-r8 | Same as B3b | Rank 8 | 0.355 | – | 0.919 | – | 0.082 | – |
| 11 Oct | B6 | 12,773 (HPC10K Slurm data and 2,000 old examples) | Same as B2 | 0.369 | 0.427 | 0.949 | 0.923 | 0.098 | – |
| 12 Oct | B7 | 20,472 (B6 data and new data for weak topics; v3 source text removed) | Same as B2 | 0.398 | 0.450 * | 0.939 | 0.945 * | 0.085 | 0.036 * |
| 13 Oct | Teacher model | – | – | judge running | judge running | – | – | – | – |
| 13 Oct | B7 with 30/50/70% of its changes | Same as B7 | No training | judge running | judge running | – | – | – | – |
| 13 Oct | B8a | Same as B7 | Learning rate 2e-5, rank 8, half an epoch | – | training | – | – | – | – |

\* 2 of 3 seeds. The judge step for the third seed runs again.

- The v2 test set has 66 questions on topics that we do not train. The margin of error is ±0.12.
- The v3 test set has 200 questions on topics that we train.
- B1 to B3b-r8 have no v3 result, because the v3 test set did not exist then.
- For B6, the v3 result is a development result: B6 trained on 910 examples from the v3 source text.

Important data steps:

1. **B1:** Only 0.8% of the B1 data was about Slurm. Thus, we made the HPC10K data. The teacher model generated 14,670 examples from 34 documents. We kept 10,773 examples.
2. **Reviewer problem:** 47% of the reviews were not valid. The reviewer wrote spaces until the token limit. We added a JSON format limit (xgrammar `disable_any_whitespace`). Now approximately 2.6% of the reviews are not valid.
3. **Data for weak topics:** The teacher model generated 18,988 examples. We kept 10,248 examples. 4,352 examples wait for a review. 1,084 of these examples did not get a reply from the reviewer.
4. **v3 test set:** Vaibhav made the v3 test set (PR #36). We removed the source text of the v3 questions from the training data. This removed 910 HPC10K examples and 1,300 new examples.
5. **Tool use:** Since B3b, the tool-use results of the student model are correct. The decision accuracy is 1.0. The student model makes 0 incorrect tool calls. The base model makes incorrect tool calls in 17% of the cases.

## 4. Why the student model loses

The student model loses because of missing facts and incorrect facts. The judge is not the cause. The data below is from B7 on the v2 test set.

### 4.1 The test facts are not in the training data

This table shows the number of training examples (of 20,472) that contain the key terms of the 31 questions that the student model loses most frequently.

| Key term | Training examples |
|---|---|
| `SLURM_ARRAY_JOB_ID`, `--no-requeue`, `USR1` / `SIGUSR1` | 0 |
| `--signal` | 1 |
| `--requeue` | 2 |
| `--array`, `SLURM_ARRAY_TASK_ID` | 3 |
| `--parsable` | 39 |
| `--dependency` | 180 |
| `afterok` | 273 |

The first 5 rows have 0 or almost 0 examples. This is intentional. The `heldout_pattern` setting in `data/sources/hpc10k_sources.json` removes all text about arrays, requeue, signals and preemption.

### 4.2 The judge is correct

- When the student model loses both comparisons, its answer has 0.62 of the key terms of the reference answer. The base model answer has 0.73.
- When the student model wins both comparisons, its answer has 0.76 of the key terms. The base model answer has 0.67.
- The answer length does not have a constant relation to the result.

### 4.3 Fine-tuning changed the answer style

- Median answer length: student model 313 characters, base model 1,381 characters, training answers 442 characters.
- The system prompt tells the model to write concise answers.
- Example (question arr-009, "Get the job ID from sbatch"): The student model gave `$SBATCH_JOB_ID` as the answer. This variable does not exist. The base model gave 3 correct methods and won.
- A published study gives the same result. When you fine-tune a model on facts that it does not know, the model invents more facts (Gekhman et al. 2024).

### 4.4 Results for each pair (mean number of questions of 66)

| Round | Student model wins 2 times | Base model wins 2 times | Split or tie |
|---|---|---|---|
| B6 | 15.7 | 32.0 | 18.3 |
| B7 | 17.3 | 30.7 | 18.0 |

The weakest question types (B7 score; 0.5 is equal):

- Array procedures: 0.21.
- Requeue problem analysis: 0.26.
- Requeue scripts: 0.31.
- Concept questions: 0.49 to 0.50.

### 4.5 The student model also loses on trained topics (v3)

Section 3.1 gives the v3 results. B6 has 0.427 and B7 has 0.450. Both lose against the base model.

- B7 is better than B6 by approximately 0.02. This difference is small.
- The B7 bad-flag rate on v3 (0.036) is lower than on v2 (0.085). It is still higher than the base model (0.021).
- The B7 check pass rate on v3 (0.945) is higher than the base model (0.885). But the judge prefers the base model answers more frequently.

### 4.6 B6 topics on the v3 test set

| v3 topic | Questions | Student model score (0.5 is equal) |
|---|---|---|
| Scripts with many limits | 20 | 0.74 |
| Resources | 30 | 0.53 |
| Job control | 25 | 0.50 |
| Queue and cancel | 20 | 0.43 |
| Accounting | 30 | 0.41 |
| Dependencies | 30 | 0.36 |
| Problem analysis | 20 | 0.33 |
| Output and mail | 25 | 0.16 |

- The student model wins on job scripts with many limits.
- The student model loses on output and mail options, problem analysis and dependencies. These are the same topics where it invents flags.
- The check pass rate of the student model is higher than that of the base model (0.923 and 0.885). But the student model does not win more comparisons with the judge.
- B6 trained on 910 examples from the v3 source text before we removed it. Thus, for B6 the v3 result is a development result, not a blind test. B7 does not have this problem.

## 5. Problems that slow us

### 5.1 Evaluation

- Until 13 October, the main result was only about topics that we do not train. Thus, the result did not show if the student model learned the trained topics.
- Now the v3 test set is the main result for training runs. The v2 test set is a check: the student model must not become worse on topics that we do not train.
- 66 questions give a margin of ±0.12. The differences between rounds on v2 are less than this margin.
- The v3 test set has 200 questions. The results of different seeds are in a range of approximately ±0.02.
- We do not have a result of the teacher model against the base model.

### 5.2 Training method

- We use only fine-tuning on the text of the teacher model.
- We do not use preference training.
- We do not use the token probabilities of the teacher model. The teacher model gives the top 20 token probabilities (`top_logprobs: 20`), but the training code does not use them.
- The training and the evaluation use the same "concise" system prompt.
- The validation loss stops decreasing at approximately half of the epoch (B6: at step 500).

### 5.3 Compute time

- Each training seed takes 2.5 hours for 20,000 examples (5 to 6 seconds for each step).
- These items cause the time: the 4-bit weights change back to full size at each step, DoRA does more work for each layer, gradient checkpointing, and a full validation run every 100 steps.
- The validation set contains full source documents. Thus, its size changes with the seed (1,038 to 2,867 examples). The size of the training data also changes.
- Data generation takes 5 to 6 hours for each part. 4,352 examples wait for a review.

### 5.4 Engineering

- Fixed on 13 October: `slurm/env.sh` now uses the account `EUHPC_D30_031`. It also accepts a personal `VENV` path. Before this fix, the first evaluation jobs of 13 October failed.
- Fixed on 13 October: we merged the v3 test set (PR #36) into our branch. Before this merge, the scripts used 2 different source trees.
- The v3 source check used the current directory to find `data/eval`. From an incorrect directory, it removed 0 examples and gave no error. PR #36 has the fix.

## 6. Jobs on the `s_tra_ncc` reservation (13 October)

| Job | Question | Status at 11:30 |
|---|---|---|
| B7 on v3, 3 seeds (60276546) | Does the larger B7 data win on trained topics? (B6: 0.427) | 2 of 3 seeds complete (mean 0.450); the third seed runs again (judge) |
| B6 on v3, 3 seeds (60276547) | Does our run give the same result as the run of Vaibhav? | Complete: 0.421 (Vaibhav: 0.427) |
| Teacher model answers (60276548) | Does the teacher model win against the base model? | Answers complete |
| B7 with 30%, 50% and 70% of its changes (60276549) | Does a smaller change keep the gains and lose less base knowledge? | Answers complete |
| Judge for the teacher model and the 30/50/70% answers (60278960) | Win rates on v2 and v3 for these 4 models | Running (8 comparisons) |
| B8a, 3 seeds (60276743) | Learning rate 2e-5, rank 8, half an epoch, same data, scored on v3 | Training; evaluation waits |

## 7. Options for the next steps

The table shows the options from the lowest cost to the highest cost. Options 1 to 4 are in progress.

| # | Option | Problem that it solves | Cost | Status or notes |
|---|---|---|---|---|
| 1 | Score on trained topics (v3) | Shows if the training method operates | 1 GPU hour | B6: 0.427; B7: 0.450. Both lose |
| 2 | Teacher model against base model | Shows the best possible result | 1 GPU hour | Answers complete; judge running |
| 3 | Use only 30% to 70% of the trained change | Loss of base knowledge | No training | Answers complete; judge running |
| 4 | Gentle training (low learning rate, rank 8, early stop) | Loss of base knowledge; invented facts | 1.5 hours | Running as B8a |
| 5 | Remove "concise"; the teacher model writes full answers | Short answers with only 1 method | New data run | Not started |
| 6 | Add 20% to 50% general data or base model answers | Loss of base knowledge | Small | Standard method |
| 7 | Preference training (DPO); rejected answers contain invented flags | Invented flags | 1 day | Our flag checker finds the pairs; TRL `DPOTrainer` |
| 8 | On-policy distillation (GKD, reverse KL) | The student model is certain where the teacher model is not | 1 to 2 days | The teacher model must run during training; TRL `GKDTrainer` (experimental) |
| 9 | Distillation from the teacher token probabilities | More information for each example | 1 day | arcee-ai DistillKit |
| 10 | Search the Slurm documents during the answer (RAG) | Missing facts on all topics | 1 to 2 days | Best solution for topics that we do not train; runs on a laptop |
| 11 | Larger student model (Qwen3-8B) | More knowledge before training | New training | Approximately 5 GB with 4-bit weights |

A different training framework (Unsloth, Axolotl, LLaMA-Factory) can make the training faster or easier. But it does not change the win rate. The evaluation, the data and the training method change the win rate.

Our plan:

1. B6 (0.427) and B7 (0.450) lose on trained topics. More data gives only a small improvement. Thus, we must change the training method.
2. Next, we will do option 7 (DPO) and options 5 and 6 (full answers, mixed data).
3. These options will target output and mail, problem analysis and dependencies.
4. If the teacher model does not win against the base model, we will change our main work to option 10 (RAG).

## 8. Questions for the mentor

1. Our new goal: the student model must win on trained topics (v3) and must not become worse on topics that we do not train (v2). The old goal was a win on topics that we do not train. Do you agree with the new goal?
2. Can we use RAG on the Slurm documents in the hackathon? Or must the improvement come only from the model?
3. We have few days. Which method do you recommend: DPO against invented flags, or on-policy distillation?
4. The v2 test set has 66 questions and a margin of ±0.12. Must we make it larger before we compare rounds?
5. How can we make QDoRA training faster on A100 GPUs? For example: bf16 LoRA without 4-bit weights, sequence packing, or fewer validation runs.

## Sources

- Gekhman et al. 2024, "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" — https://arxiv.org/abs/2405.05904
- Biderman et al. 2024, "LoRA Learns Less and Forgets Less" — https://arxiv.org/abs/2405.09673
- Agarwal et al. 2024, "On-Policy Distillation of Language Models" (GKD) — https://arxiv.org/abs/2306.13649
- Gu et al. 2024, "MiniLLM" — https://arxiv.org/abs/2306.08543
- Rafailov et al. 2023, "Direct Preference Optimization" — https://arxiv.org/abs/2305.18290
- Wortsman et al. 2022, "Robust fine-tuning of zero-shot models" (WiSE-FT) — https://arxiv.org/abs/2109.01903
- Ovadia et al. 2024, "Fine-Tuning or Retrieval?" — https://arxiv.org/abs/2312.05934
- TRL trainers — https://huggingface.co/docs/trl ; arcee-ai DistillKit — https://github.com/arcee-ai/DistillKit
- Project files: `docs/TRAINING_PROGRESS.md`, `docs/plan/HPC10K_DATA.md`, `configs/qwen3_4b_qdora_b2.yaml`, `src/distillkit/train.py`, `src/distillkit/evaluate.py`, `slurm/`
