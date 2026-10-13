# DistillKit: Status Report for the Mentor (Version 2)

Date: 13 October 2026, 15:40. Team: Alembic. Mentor: Simeon Harrison.

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

Our target is a win rate of more than 0.5 against the base model. On 13 October, 2 student models won on the v3 test set (trained topics) for the mean of 3 seeds. No student model wins on the v2 test set (topics that we do not train).

- On the v2 test set, B7 has a win rate of 0.398.
- On the v3 test set, B7 has a win rate of 0.438.
- With only 20% of the B7 changes, the student model wins on the v3 test set: 0.528 (mean of 3 seeds). This is the first result above 0.5. On v2, it has 0.481, which is better than B7 (0.398) but less than 0.5.
- D3 (training with a KL penalty to the base model) is the first trained model that wins on v3: 0.525 (3 seeds). On v2 it has 0.442.
- DPO on x0.2 (D5) makes the result worse: 0.442 on v2 and 0.42 on v3.
- The teacher model wins against the base model: 0.689 on v2 and 0.585 on v3.
- Section 3 shows all results in one table. All values are means of the seeds.
- From B1 to B7, the v2 win rate increased from 0.324 to 0.398.
- The increase from B6 (0.369) to B7 (0.398) is less than the margin of error (±0.12).

We found two causes:

1. The v2 test set measures topics that we do not train. The student model cannot learn these topics from our data.
2. Full fine-tuning (B7) removes facts that the base model knows. The key-term coverage on v3 drops from 0.61 (base) to 0.53 (B7). The models that win (x0.2 and D3) keep the base coverage.
3. The win does not come from longer answers. In each model, the answer length does not predict the judge score. The key-term coverage predicts it.

We do not have these results yet:

- The D5 v3 result for seeds 1 and 2 (1 invalid judge verdict in each).


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
| 12 Oct | B7 | 20,472 (B6 data and new data for weak topics; v3 source text removed) | Same as B2 | 0.398 | 0.438 | 0.939 | 0.958 | 0.085 | 0.027 |
| 13 Oct | Teacher model (Qwen3-32B) | – | – | **0.689** | **0.585** | 0.985 | 0.945 | 0.006 | 0.015 |
| 13 Oct | B7 with 10% of its changes † | Same as B7 | No training | 0.481 | **0.522** | – | – | – | – |
| 13 Oct | B7 with 20% of its changes † | Same as B7 | No training | 0.481 | **0.528** | – | – | – | – |
| 13 Oct | B7 with 30% of its changes † | Same as B7 | No training | 0.443 | **0.510** | – | 0.940 | – | 0.020 |
| 13 Oct | B7 with 40% of its changes † | Same as B7 | No training | 0.424 | 0.495 | – | – | – | – |
| 13 Oct | B7 with 50% of its changes † | Same as B7 | No training | 0.436 (1) | 0.476 (1) | – | 0.940 | – | 0.027 |
| 13 Oct | B7 with 70% of its changes † | Same as B7 | No training | 0.409 (1) | 0.418 (1) | – | 0.940 | – | 0.031 |
| 13 Oct | B7, early stop at step 300 / 500 (D4) | Same as B7 | Fewer steps | 0.374 / 0.396 | 0.406 / 0.407 | – | – | – | – |
| 13 Oct | B8a | Same as B7 | Learning rate 2e-5, rank 8, half an epoch | – | 0.347 | – | – | – | – |
| 13 Oct | D0 | 12,892 good-quality B7 examples | bf16 weights (control) | 0.376 | 0.438 | – | – | – | – |
| 13 Oct | D2 | Same as D0 | 0.5 × CE + 0.5 × KL to the teacher top-20 tokens | 0.361 | 0.450 | – | – | – | – |
| 13 Oct | **D3** | Same as D0 | CE + 0.1 × KL to the base model | 0.442 | **0.525** | – | – | – | – |
| 13 Oct | D5 | DPO pairs (teacher vs student) | DPO on x0.2, beta 0.1 | 0.442 | 0.42 (1) | – | – | – | – |

† We did not train again. We used the weights W = base + s × (B7 − base), for s = 0.1 to 0.7. The number in brackets is the number of seeds with a result, if it is less than 3. The v3 scale choice is a development result.

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

### 3.2 Head-to-head check: x0.2 against the base model (20 questions)

One reviewer compared 20 random answer pairs (10 v3, 10 v2, seed 42) with the reference answers.
Details for each question: [HEAD_TO_HEAD_x0.2.md](HEAD_TO_HEAD_x0.2.md).

| Result | Base model | x0.2 |
|---|---|---|
| Correct | 6 | 5 |
| Partly correct | 9 | 10 |
| Wrong | 5 | 5 |
| Better answer | 7 | 3 |
| Same quality | 10 | 10 |
| x0.2 win rate: manual check / judge | – | 0.40 / 0.43 |

- Both models make the same types of errors. Neither model knows `--parsable` or `afternotok`.
- x0.2 wins when the base model adds incorrect details. It loses when it explains a mechanism incorrectly.
- 20 questions cannot confirm a difference as small as 0.528 against 0.5.

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

Section 3.1 gives the v3 results. B6 has 0.427 and B7 has 0.438. Both lose against the base model.

- B7 is better than B6 by approximately 0.01. This difference is small.
- The B7 bad-flag rate on v3 (0.027) is lower than on v2 (0.085). It is still higher than the base model (0.021).
- The B7 check pass rate on v3 (0.958) is higher than the base model (0.885). But the judge prefers the base model answers more frequently.

### 4.6 A smaller change wins, and the teacher model wins

- The teacher model wins against the base model on both test sets (0.689 and 0.585). Thus, a better model from distillation is possible.
- When we use less of the B7 change, the result improves. On v3 (3 seeds): 40% gives 0.495, 30% gives 0.510, 20% gives 0.528 and 10% gives 0.522.
- This agrees with the diagnosis in sections 4.1 to 4.3. The full training removes too much knowledge of the base model. A small part of the change keeps the gains and keeps the base knowledge.
- D3 gets a similar result with training: a KL penalty to the base model keeps the base knowledge (v3 0.525, v2 0.442).

### 4.8 Answer length and key terms (all scaled models, D4 and D5)

| Model | v3 win rate | Median length, student / base | Key-term coverage, student / base |
|---|---|---|---|
| x0.1 | 0.522 | 1,526 / 1,619 | 0.60 / 0.61 |
| x0.2 | 0.528 | 1,220 / 1,619 | 0.61 / 0.61 |
| x0.3 | 0.510 | 886 / 1,619 | 0.60 / 0.61 |
| B7 | 0.438 | 365 / 1,619 | 0.53 / 0.61 |
| D4 step 300 | 0.406 | 378 / 1,619 | 0.52 / 0.61 |
| D5 (DPO) | 0.421 | 442 / 1,619 | 0.53 / 0.61 |

- In each model, the length does not predict the score (Spearman rho -0.10 to 0.08).
- The coverage difference predicts the score (rho 0.14 to 0.42), also when the length is fixed.
- Script: `research/length_analysis/length_analysis.py`. Full tables: `docs/plan/STUDENT_DISTILL.md`.

### 4.7 B6 topics on the v3 test set

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
- The teacher model wins against the base model (v2 0.689, v3 0.585).

### 5.2 Training method

- Until 13 October, we used only fine-tuning on the text of the teacher model.
- On 13 October, we tested 3 new methods. Logit distillation from the teacher top-20 token probabilities (D2) did not help. Preference training (DPO, D5) made the result worse. A KL penalty to the base model (D3) wins on v3.
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
- Changed on 13 October: training jobs for each seed now use 1 full node (4 GPUs, 480 GB) with 1 GPU for each seed. Other jobs use 120 GB for each GPU. This operated correctly for D0, D2 and D3 on 13 October (58 to 67 minutes for 12,892 examples).
- Changed on 13 October: new training runs use DoRA with bf16 base weights, not 4-bit weights (`configs/qwen3_4b_dora_bf16.yaml`). The A100 GPU has enough memory. 4-bit weights make each step slower, and the adapter trains on weights that are different from the weights that it uses at evaluation.
- Fixed on 13 October: `generate.sbatch` now removes questions of all test files from new training data, not only the questions of the config test file.
- The v3 source check used the current directory to find `data/eval`. From an incorrect directory, it removed 0 examples and gave no error. PR #36 has the fix.

## 6. Jobs on the `s_tra_ncc` reservation (13 October)

| Job | Question | Status at 15:32 |
|---|---|---|
| B7 on v3, 3 seeds (60276546, 60279954) | Does the larger B7 data win on trained topics? (B6: 0.427) | Complete: 0.438 |
| B6 on v3, 3 seeds (60276547) | Does our run give the same result as the run of Vaibhav? | Complete: 0.421 (Vaibhav: 0.427) |
| Teacher model answers (60276548) | Does the teacher model win against the base model? | Complete: yes (v2 0.689, v3 0.585) |
| B7 with 30%, 50% and 70% of its changes (60276549) | Does a smaller change keep the gains and lose less base knowledge? | Answers complete |
| Judge for the teacher model and the 30/50/70% answers (60278960) | Win rates on v2 and v3 for these 4 models | Complete |
| 20%, 30% and 40% of the changes for seeds 1, 2 and 42 (60281969 to 60281971), judge (60281973) | Is the 0.526 result real for all seeds? | Complete: x0.2 0.528, x0.3 0.510, x0.4 0.495 on v3 |
| DPO pair mining (60284208) | Pairs for preference training | Complete: 858 pairs (303 invented flags, 555 missing flags) from 6,000 questions |
| B8a, 3 seeds (60276743, 60276744) | Learning rate 2e-5, rank 8, half an epoch, same data, scored on v3 | Complete: 0.347 on v3. Gentle training loses more |
| x0.1 answers (60294836/842/850), D4 early stop answers (60294853/857/859), judge (60318026) | Does a smaller change or an early stop help? | x0.1: 0.481 v2, 0.522 v3. Early stop loses (0.37 to 0.41). Complete |
| D0 / D2 / D3 training (60302556/558/559), evaluation v3 (60322950/954/957) and v2 (60323699/702/705) | Control, logit distillation (KD) and base-KL training on the good-quality data | Complete: D0 0.376 / 0.438, D2 0.361 / 0.450, D3 0.442 / **0.525** (v2 / v3) |
| D5: DPO on x0.2 (60302560, 579, 582), judge (60322599) | Does DPO remove invented flags and keep the win? | Complete: v2 0.442, v3 0.42 (1 seed). Worse than x0.2 |

## 7. Options for the next steps

The table shows the options from the lowest cost to the highest cost. Options 1 to 4, 7 and 9 are complete.

| # | Option | Problem that it solves | Cost | Status or notes |
|---|---|---|---|---|
| 1 | Score on trained topics (v3) | Shows if the training method operates | 1 GPU hour | B6: 0.427; B7: 0.438. Both lose |
| 2 | Teacher model against base model | Shows the best possible result | 1 GPU hour | Complete: the teacher model wins (0.689, 0.585) |
| 3 | Use only 20% to 70% of the trained change | Loss of base knowledge | No training | 20% wins on v3 (0.528, 3 seeds); 30% 0.510 |
| 4 | Gentle training (low learning rate, rank 8, early stop) | Loss of base knowledge; invented facts | 1.5 hours | B8a: 0.347 on v3; early stop (D4): 0.40 to 0.41. Both lose |
| 5 | Remove "concise"; the teacher model writes full answers | Short answers with only 1 method | New data run | Not started |
| 6 | Add 20% to 50% general data or base model answers | Loss of base knowledge | Small | Standard method. D3 (KL to the base model) solves the same problem and wins on v3 (0.525) |
| 7 | Preference training (DPO); rejected answers contain invented flags | Invented flags | 1 day | D5: worse than x0.2 (v2 0.442, v3 0.42). Stop |
| 8 | On-policy distillation (GKD, reverse KL) | The student model is certain where the teacher model is not | 1 to 2 days | The teacher model must run during training; TRL `GKDTrainer` (experimental) |
| 9 | Distillation from the teacher token probabilities | More information for each example | 1 day | D2: v2 0.361, v3 0.450. Almost equal to the control (D0). Does not help |
| 10 | Search the Slurm documents during the answer (RAG) | Missing facts on all topics | 1 to 2 days | Best solution for topics that we do not train; runs on a laptop |
| 11 | Larger student model (Qwen3-8B) | More knowledge before training | New training | Approximately 16 GB with bf16 weights; fits on 1 A100 |

A different training framework (Unsloth, Axolotl, LLaMA-Factory) can make the training faster or easier. But it does not change the win rate. The evaluation, the data and the training method change the win rate.

Our plan:

1. B6 (0.427) and B7 (0.438) lose on trained topics. More data gives only a small improvement. Thus, we must change the training method.
2. 20% of the B7 change wins on v3 for the mean of 3 seeds (0.528). This is our first model that wins. The length analysis shows that the win does not come from longer answers. In each model, the length does not predict the score (rho -0.10 to 0.08). The key-term coverage predicts it (rho 0.14 to 0.42). x0.2 keeps the base facts (coverage 0.61, equal to the base model). Full training (B7) loses them (0.53).
3. D3 (KL penalty to the base model) wins on v3 (0.525) and is the best trained model. D2 (logit distillation) and D5 (DPO) do not help. Next: a scaled merge of D3 (20% to 50%), and options 5 and 6 (full answers, mixed data).
4. These options will target output and mail, problem analysis and dependencies.
5. The teacher model wins, so distillation can operate. RAG (option 10) stays the best solution for topics that we do not train.

## 8. Questions for the mentor

1. Our new goal: the student model must win on trained topics (v3) and must not become worse on topics that we do not train (v2). The old goal was a win on topics that we do not train. Do you agree with the new goal?
2. Can we use RAG on the Slurm documents in the hackathon? Or must the improvement come only from the model?
3. DPO made the result worse, and D3 (KL to the base model) wins on v3. Must we continue with D3 (for example a scaled D3), or test on-policy distillation?
4. The v2 test set has 66 questions and a margin of ±0.12. Must we make it larger before we compare rounds?
5. We changed to DoRA with bf16 weights and 1 GPU for each seed. Is data parallel training on 4 GPUs for each seed a better use of the nodes? Is DoRA better than LoRA for our task?
6. Is a scaled model (x0.2, v3 0.528) an acceptable final model, or must the improvement come from the training (D3, v3 0.525)?

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
