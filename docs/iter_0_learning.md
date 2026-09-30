# Iteration 0: first end-to-end toy run (2026-09-30)

*What we ran, what broke, what we learned. Written in plain language for anyone onboarding; see [training.md](training.md) for how training works.*

**TL;DR:** The whole pipeline (generate → verify → train → evaluate) now runs end to end on the laptop. But the trained student is **worse** than the base model it started from: it wins only **27.5%** of judge comparisons (target: > 60%), because it **gets stuck repeating itself and never stops**. The training data was checked and is built correctly; the most likely cause is training settings that are too aggressive for a 0.6B model on 40 samples. That is a hypothesis, not yet tested.

## Setup

| | |
|---|---|
| Student | `Qwen/Qwen3-0.6B` + LoRA (rank 64, alpha 128, all 7 linear layer types) |
| Teacher | `qwen3.8-flash` (OpenCode Zen API) |
| Judge | `deepseek-v4.1-flash` (different family from the teacher) |
| Training data | 20 questions × 2 teacher answers = 40 samples, from 4 hand-written Slurm docs |
| Eval set | 10 hand-written questions from a 5th doc (`slurm_job_arrays`) held out from training |
| Hardware | RTX 2060 6 GB, fp16 |
| Config | `configs/toy.yaml` |

## Stage by stage

### 1. `generate`: works, but slow
- 40 answers, none empty, average ~890 characters. Answers are grounded in the source doc and use fenced scripts.
- **7m38s for 60 API calls** (~30 s per call at concurrency 4). Too slow for the planned ~5,000 samples; needs higher concurrency.

### 2. `verify`: all 4 rejections were checker bugs
The first run kept 36/40. All 4 rejected answers were actually fine:

| Rejection | Real cause | Fix |
|---|---|---|
| 2 × "bad flag" | `srun --jobid` is a real flag, missing from our hand-written `slurm_flags.txt` | Added `jobid` |
| 2 × "bash syntax error" | Placeholders like `<jobid>` look like file redirections to `bash -n` | Replace `<placeholder>` with a dummy word before checking |

After the fixes: **40/40 kept**.

**Learning:** always read the rejected rows. A checker written from memory rejects good data silently.

### 3. `train`: took three attempts, then worked
| Attempt | Failure | Fix |
|---|---|---|
| 1 | transformers 5 no longer accepts `warmup_ratio` | `warmup_steps=0.03` (a float < 1 is read as a ratio) |
| 2 | torch 2.14 tries to compile a Triton GPU kernel, which needs a C compiler this machine doesn't have | `TORCH_DISABLE_NATIVE_JIT=1`, set in `src/distillkit/cli.py` |
| 3 | ✅ | — |

Result: 6 optimizer steps in 28 seconds. Loss **1.85 → 1.33**, token accuracy 59% → 69%. Adapter saved (162 MB; 40.4M trainable numbers = 6.3% of the model).

**Learning:** falling loss only proves the pipeline runs. A loss of 1.33 on 40 samples means the model did not even memorise them.

### 4. `evaluate`: two judge bugs, then a real (bad) result

**Bug 1, found by reading the code before running:** the judge's verdict was read as the *first character* of its reply. A reply like "Answer: B" would have counted as "A", and anything unreadable silently became a tie. Fixed: take the **last** standalone A/B/T, save the judge's raw replies, and report a `judge_unparsed` count (should be 0).

**Bug 2, found by that new count:** the first run had **13 of 20 judge replies empty**. The judge reasons before it answers; comparing two long answers takes ~1,000 tokens of reasoning, and our limit was 512, so it was cut off before giving a verdict. Its reported 32.5% win rate was meaningless. Fixed: judge `max_tokens` 512 → 4096.

Also changed: the 20 judge calls now run in parallel (4 at a time) instead of one by one. Full evaluate run: **3m11s**.

**Learning:** an LLM judge can fail silently. Always log its raw replies and count the ones you couldn't parse.

## The result

| | Base (no training) | Student (after training) |
|---|---|---|
| **Judge win rate, student vs base** | — | **27.5%** (target > 60%) |
| Answers that hit the 512-token limit | **0/10** | **9/10** |
| Answers passing all flag + bash checks | 8/10 | 7/10 |
| Answers with made-up flags | 1 (`--job-id`) | 3 (`--output-file`, `--output-format`, `--cpus_per_task`) |
| Judge replies unparsed | | 1 of 20 |

Per question, each pair was judged twice with A/B swapped:

| Outcome | Questions |
|---|---|
| Base wins both orders | 6 (Q1, Q2, Q5, Q6, Q7, Q8) |
| Student wins both orders | 1 (Q9) |
| Student wins once, tie once | 1 (Q0) |
| Split: each wins when shown first (judge prefers position A) | 2 (Q3, Q4) |

### What the student's answers look like

**Q7: "What does the singleton dependency do?"**
- Reference: *makes the job wait until any earlier jobs with the same job name and user have finished.*
- Base (wrong, but coherent): *"…ensures a single instance of a dependency is used across all processes…"*
- Student (wrong and looping): *"The singleton dependency is a dependency that ensures that the Slurm cluster is always available and that the Slurm cluster is always available. It is used to ensure that the Slurm cluster is always available and that the Slurm cluster is always available."*

**Q2: "How do I name the output files of an array job so each task writes to its own log file?"**
- Reference: `--output=logs/%x_%A_%a.out` (`%A` = array job ID, `%a` = task index).
- Base: wrong (one `#SBATCH --output` line per task), but short.
- Student: repeats the same sentence plus a command with invented flags (`srun --output=myjob.out --output-format=text --output-file=myjob.out`) until it hits the 512-token limit.

## Diagnosis: why does the student never stop?

### Ruled out: broken training data
We inspected exactly what the trainer feeds the model:
- The prompt (114 tokens in the checked sample) is excluded from the loss; learning starts at the first token of the answer. ✅
- The stop token `<|im_end|>` is included in what the model learns to produce, exactly once per sample. ✅
- Training and generation use the same stop token. ✅

### Quick experiments (local GPU, no API)

| Test | Answers hitting 512 tokens |
|---|---|
| Student, **training** questions, greedy | **5/5**, doesn't stop even on data it was trained on |
| Student, eval questions, sampling (temperature 0.7, top-p 0.8, top-k 20) | 7/10 |
| Base, eval questions, sampling | 0/10 |

So the looping is **caused by training**, not by the decoding method or the eval questions.

### Most likely causes (hypotheses, not yet tested)
1. **Training too aggressive for this size.** Learning rate 2e-4 with rank-64 LoRA on every layer (scale ×2) moves a 0.6B model a lot in a few steps. It picked up the teacher's *long style* but not the coherence to finish.
2. **Teacher answers are long for a 0.6B student:** median 245 tokens, max 909 answer tokens (vs. 47–290 tokens for the base model's own answers). Small models lose the thread in long answers and start looping. `PLAN.md` already says answers should be "short & direct"; the teacher prompt doesn't enforce it yet.
3. **Less likely:** training ran with fp32 weights but eval loads the model in fp16.

## Next experiments (in order)

1. **Gentler training:** learning rate 5e-5, LoRA rank 16. Save each attempt to its own folder. First check only how many answers hit 512 tokens (cheap); run the full judge only if the looping is fixed.
2. If still looping: evaluate in fp32 to rule out cause 3.
3. **Shorter teacher answers:** add a length limit and a "short & direct" instruction to the generation prompt.
4. More data (40 samples is tiny) and ~50 eval questions, so a win rate means something.

## Other open issues found in this iteration
- **Flag checker reports an empty flag (`''`)** on one student answer, probably a bare `--` token. Small checker bug.
- **Judge position bias** visible on 2/10 questions (whichever answer is shown first wins). Judging in both orders cancels it, which is why we do it.
- **1 judge reply was still empty** even with a 4,096-token limit. Rare, but `judge_unparsed` should be watched.
- **The flag list is still hand-written from memory.** Rebuild from the real `man sbatch`/`srun`/`salloc` pages (planned by Oct 5).
- **The base model is also weak** on these questions (both of its answers above are wrong). That's good news: there is plenty of room for a properly trained student to win.

## Lessons for the next iteration
1. Read rejected rows and judge replies by hand; don't trust counts alone.
2. Every automatic check needs a "couldn't decide" counter (`judge_unparsed`, answers hitting the length limit).
3. Check answer **length** and **stopping** before paying for a judge run. Looping is visible for free.
4. Loss going down ≠ better model. Only the held-out eval tells you that.
5. Library versions move fast (transformers 5, torch 2.14): expect small breakages on first runs.
