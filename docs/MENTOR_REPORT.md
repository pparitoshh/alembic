# DistillKit: status report for the mentor

As of 13 October 2026, 11:00. Team Alembic. Mentor: Simeon Harrison.

## 1. Where we stand

Our fine-tuned 4B student still loses to the base 4B model it started from. It wins **0.398** of head-to-head
comparisons (B7, mean of 3 seeds), so the base model effectively scores 0.60. Our target is above 0.5.

Seven training rounds have moved the student only from 0.324 (B1) to 0.398 (B7). The last step, from B6 (0.369) to B7,
is smaller than the ±0.12 margin of error.

Our main diagnosis comes from how the test was designed. All 66 test questions are about two topics that we
deliberately kept out of training: job arrays and requeue/signals. Fine-tuning made the student shorter and more
confident. On topics it was never taught, that confidence shows up as invented `sbatch` flags and wrong facts.

Two things we have never measured, both running today (section 6):

- whether the student beats the base on topics it **was** trained on (eval v3, 200 questions);
- whether the teacher itself beats the base. If it doesn't, copying the teacher cannot get the student above 0.5.

## 2. How the system works

```
 Slurm / HPC docs ──► Teacher writes Q&A ──► Reviewer checks ──► Verify + decontam ──► Train student ──► Judge vs base
 (34 docs; test     (Qwen3-32B-AWQ,        (Gemma-4-26B,       (flag list, bash      (Qwen3-4B,         (gpt-oss-20b,
  topics removed)    vLLM, 1 A100)          same job)           syntax, dedup,        QDoRA, 1 A100      both answer
                                                                 eval-overlap guard)   per seed, 3 seeds) orders)
```

| Stage | Model | Hardware | Typical time |
|---|---|---|---|
| Generate + review + verify | Qwen3-32B-AWQ teacher, Gemma-4-26B reviewer | 1 A100 per shard | 5–6 h per shard (HPC10K, 5 shards) |
| Train | Qwen3-4B-Instruct-2507, QDoRA (4-bit DoRA, rank 16) | 1 A100 per seed | 2h23–2h46 for B7 (20k records) |
| Answer + judge | base + student (HF greedy, 512 tokens), gpt-oss-20b judge | 1 A100 per seed | about 20 min (66 questions) |

Training settings (B2–B7): learning rate 5e-5, 1 epoch, cosine schedule, effective batch 16, max length 2048, loss on
assistant turns only.

Scoring: each question is judged twice, once with each answer shown first. A win counts 1, a tie 0.5 and a loss 0.
The judge sees a reference answer and is told to "penalise invented options heavily; do not reward length".

## 3. What we did

| Date | Round | Data | Change | Win rate vs base (mean) | Bad-flag rate |
|---|---|---|---|---|---|
| Oct 10 | B1 | 4,775 generic HPC records | lr 1e-4, 3 epochs | 0.324 | 0.289 |
| Oct 10 | B2 | same | lr 5e-5, 1 epoch | 0.326 | 0.220 |
| Oct 10 | B3b | + 182 Slurm prose and tool records (×3) | same as B2 | 0.359 | 0.059 |
| Oct 10 | B3b-r8 | same | rank 8 | 0.355 | 0.082 |
| Oct 11 | B6 | 12,773 (HPC10K Slurm data + 2k retained) | same as B2 | 0.369 | 0.098 |
| Oct 12 | B7 | 20,472 (+ weak-area top-up, eval v3 overlap removed) | same as B2 | 0.398 | 0.085 |
| — | Base | — | — | 0.50 | 0.025 |

"Bad-flag rate" is the share of `sbatch` flags in answers that don't exist.

Key data steps:

- **B1 finding:** only 0.8% of the original data was about Slurm. That led to the HPC10K pipeline: 34 Slurm and
  HPC Carpentry docs with the held-out topics removed. 14,670 records were generated and 10,773 kept.
- **Reviewer bug:** 47% of reviews were malformed, because the reviewer emitted whitespace until it hit the token
  limit. We fixed it with constrained JSON output (xgrammar `disable_any_whitespace`); about 2.6% are malformed now.
- **Weak-area top-up:** 18,988 records generated, 10,248 kept. 4,352 are still pending review, and 1,084 of those
  never got a reviewer reply.
- **Eval v3 (Vaibhav, PR #36):** 200 new questions on topics we did train on. Their source chunks are now locked out
  of training: 910 HPC10K records and 1,300 top-up records were removed.
- **Tool use is solved:** decision accuracy is 1.0 and false tool calls are 0, against 0.17 for the base model, since B3b.

B7 per seed: 0.420 / 0.383 / 0.390, check pass 0.939, bad flags 0.097 / 0.141 / 0.018.

## 4. Why the student still loses (evidence from B7)

B7 gives 198 judged pairs (3 seeds × 66 questions). The pattern in them is that the losses come from missing or wrong
facts, not from the judge.

**1. The tested facts were never in training.** The table counts training records (out of 20,472) that mention the
key terms behind the 31 questions the student loses most:

| Key term | Training records |
|---|---|
| `SLURM_ARRAY_JOB_ID`, `--no-requeue`, `USR1` / `SIGUSR1` | 0 |
| `--signal` | 1 |
| `--requeue` | 2 |
| `--array`, `SLURM_ARRAY_TASK_ID` | 3 |
| `--parsable` | 39 |
| `--dependency` | 180 |
| `afterok` | 273 |

The first five rows are zero or near zero by design: `heldout_pattern` in `data/sources/hpc10k_sources.json` drops
every block about arrays, requeue, signals and preemption.

**2. The judge looks fair; the student is wrong.**
- On questions the student loses in both orders, its answer contains 0.62 of the reference's key terms, against
  0.73 for the base.
- On questions it wins in both orders, it contains 0.76, against 0.67 for the base.
- Answer length shows no steady link to winning.

**3. Fine-tuning changed the style.**
- Median answer length: student 313 characters, base 1,381, training answers 442. The system prompt says "concisely".
- Example (arr-009, "capture the job ID from sbatch"): the student confidently answered `$SBATCH_JOB_ID`, which
  doesn't exist. The base gave three working approaches and won.
- This matches published results: fine-tuning a model on facts it doesn't already know makes it more likely to
  invent things elsewhere (Gekhman et al. 2024).

**4. Head-to-head breakdown** (questions out of 66):

| Run | Student wins both orders | Base wins both | Split or tie |
|---|---|---|---|
| B6 seed 42 / 1 / 2 | 17 / 16 / 14 | 30 / 31 / 35 | 19 / 19 / 17 |
| B7 seed 42 / 1 / 2 | 18 / 18 / 16 | 28 / 33 / 31 | 20 / 15 / 19 |

Weakest question types (B7 score, where 0.5 means even): array how-to 0.21, requeue debug 0.26, requeue script 0.31.
Concept questions are about even, at 0.49–0.50.

## 5. Bottlenecks

**Evaluation (the biggest one)**
- The only headline metric is about unseen topics. It mixes "didn't learn it" with "was never taught it".
- 66 questions give a ±0.12 margin, so the steps between rounds are within noise.
- We have no trained-topic score and no teacher-vs-base score yet.

**Method**
- Training is plain fine-tuning on teacher text only. There is no preference training and no distillation of the
  teacher's token-level probabilities.
- The teacher's top-20 token probabilities are requested (`top_logprobs: 20`), but nothing in training reads them.
- The "concise" system prompt is used in both training and evaluation.
- Validation loss flattens about halfway through the epoch (B6 from step ~500).

**Compute and time**
- Each training seed takes 2.5 h for 20k records, at 5–6 s per step. That comes from 4-bit weights being unpacked on
  every step, DoRA's extra per-layer work, gradient checkpointing, and a full validation pass every 100 steps.
- Validation holds out whole source documents, so its size swings by seed (1,038 to 2,867 records), and so does the
  amount of training data.
- Data generation is slow: 5–6 h per shard. The reviewer queue still has 4,352 records pending.

**Engineering**
- `slurm/env.sh` defaults to `SBATCH_ACCOUNT=TODO_account` and to `VENV=$REPO/.venv`. Each person and each worktree
  needs overrides; today's first eval submission failed on the venv path.
- A shared checkout plus a separate eval-v3 worktree means scripts have to be pointed at the right source tree.
- The eval-v3 source audit resolves `data/eval` relative to the current directory. Run from the wrong place, it
  silently excluded 0 records; the fix has been reported to Vaibhav.

## 6. Running today on the `s_tra_ncc` reservation

| Job | What it answers | Status (11:00) |
|---|---|---|
| B7 on eval v3, 3 seeds (60276546) | Does the student beat the base on topics it was trained on? | running |
| B6 on eval v3, 3 seeds (60276547) | Same, for the previous round | running |
| Teacher answers on both test sets (60276548) | Does the 32B teacher even beat the 4B base? (judging follows) | running |
| B7 with its learned changes at 30 / 50 / 70% (60276549) | Does a weaker adapter keep the gains but lose less of what the base knew? | running |
| B8a gentle retrain, 3 seeds (60276743) | lr 2e-5, rank 8, half an epoch, same data | training, eval queued |

## 7. Options for next steps

Ordered cheapest first. The first three are already running.

| # | Option | What it fixes | Cost | Notes |
|---|---|---|---|---|
| 1 | Score on trained topics (eval v3) | Tells us whether distillation works at all | 1 h GPU | running |
| 2 | Teacher vs base | Tells us the best result we can hope for | 1 h GPU | running |
| 3 | Scale the adapter down to 0.3–0.7 (WiSE-FT style) | Forgetting of the base model's knowledge | no training | running |
| 4 | Gentler training (low lr, rank 8, early stop) | Forgetting, overconfidence | 1.5 h | running as B8a |
| 5 | Drop "concisely"; teacher writes fuller answers | Terse, single-answer style | regenerate data | needs a new data run |
| 6 | Mix in 20–50% general data or base-model answers | Forgetting | small | standard practice |
| 7 | Preference training (DPO): rejected = answers with invented flags | Invented flags directly | 1 day | our flag checker mines the pairs for free; TRL `DPOTrainer` |
| 8 | On-policy distillation (GKD, reverse KL) | Student confident where the teacher isn't | 1–2 days | teacher must run next to the student; TRL `GKDTrainer` (experimental) |
| 9 | Distil from saved teacher token probabilities | More signal per example | 1 day | arcee-ai DistillKit; the data is already requested |
| 10 | Look up Slurm docs at answer time (RAG) | Missing facts on any topic | 1–2 days | the strongest fix for held-out topics; still runs on a laptop |
| 11 | Bigger student (Qwen3-8B) | More built-in knowledge | retrain | about 5 GB at 4-bit |

Changing the training framework itself (Unsloth, Axolotl, LLaMA-Factory) would make runs faster or simpler, but by
itself it wouldn't change the win rate. The levers that matter are the evaluation, the data and the training method.

**Our current plan:** read results 1–4 today. If the student also loses on trained topics, go to 7 (DPO) and 5/6. If
the teacher doesn't beat the base, switch the main effort to 10 (RAG).

## 8. Questions for the mentor

1. Is "beat the base on held-out topics" the right goal? Should we instead aim to win on trained topics (eval v3)
   without getting worse on held-out ones?
2. Is adding RAG over the Slurm docs acceptable within the hackathon scope, or must the gain come from the model alone?
3. With the remaining days, which method would you put effort into: DPO against invented flags, or on-policy
   distillation?
4. Would you trust a 66-question test with ±0.12 noise, or should we expand the held-out set before comparing rounds?
5. Any advice on speeding up QDoRA training on A100s? For example, bf16 LoRA without 4-bit, packing, or less frequent
   validation.

## Sources

- Gekhman et al. 2024, "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?" — https://arxiv.org/abs/2405.05904
- Biderman et al. 2024, "LoRA Learns Less and Forgets Less" — https://arxiv.org/abs/2405.09673
- Agarwal et al. 2024, "On-Policy Distillation of Language Models" (GKD) — https://arxiv.org/abs/2306.13649
- Gu et al. 2024, "MiniLLM" — https://arxiv.org/abs/2306.08543
- Rafailov et al. 2023, "Direct Preference Optimization" — https://arxiv.org/abs/2305.18290
- Wortsman et al. 2022, "Robust fine-tuning of zero-shot models" (WiSE-FT) — https://arxiv.org/abs/2109.01903
- Ovadia et al. 2024, "Fine-Tuning or Retrieval?" — https://arxiv.org/abs/2312.05934
- TRL trainers — https://huggingface.co/docs/trl ; arcee-ai DistillKit — https://github.com/arcee-ai/DistillKit
- Internal: `docs/TRAINING_PROGRESS.md`, `docs/plan/HPC10K_DATA.md`, `configs/qwen3_4b_qdora_b2.yaml`,
  `src/distillkit/{train,evaluate}.py`, `slurm/`
