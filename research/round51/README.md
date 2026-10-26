# Rounds 51 and 51b: error-focused training for GPU/MPI (Oct 26)

Goal: raise d1 (GPU/MPI/performance) towards Qwen3-235B's 0.856. Round 27 scores 0.71 there; its losses on the
requeue/signals part are wrong Slurm facts even when the right manual page is retrieved ("`--signal` does not exist",
"a requeued job gets a new job ID"). Round 50 trained round 27 on its own better answer and lost (own answers carry no
new facts), so these rounds train on answers written by a stronger model where round 27 is wrong.

## Round 51 (stopped before training)

1. `r51_build.py`: 5,000 Slurm job-control and GPU/MPI/environment questions from the verified banks (held-out
   families, near-eval questions and round 50's questions excluded), each with the app's excerpts (variant e) and its
   verified reference.
2. `r51_gen.py`, `r51_gen.sbatch`: round 27 answers each one (multi-node, resumable).
3. Qwen3-235B judges round 27's answer against the verified reference text itself.

Stopped: the judge preferred the reference on 4,589 of 5,000 questions in both orders, so the verdict does not single
out real mistakes. Judging an answer against the reference text as the competing answer is not a usable error signal.

## Round 51b

1. Round 51's questions and round 27 answers.
2. `r51b_teacher.py`, `r51b_teacher.sbatch`: Qwen3-235B answers the same prompts (same excerpts, system prompt, greedy,
   1,536 tokens).
3. Qwen3-235B judges round 27 against Qwen3-235B, with the verified reference: Qwen3-235B clearly better on 3,654.
4. `r51b_train_data.py`: target = Qwen3-235B's answer on those 3,654, round 27's own answer on 1,346 others, plus
   6,000 replay rows (Qwen3-235B distillation, KD_MIXED=1).
5. `r51b.sh` (config `r51b.env`): 3 seeds x0.5 soup into round 27, answered with variant e at 1,536 tokens, judged
   against round 27 + e.

Result (3-judge majority vs round 27): v2 +0.004, v3 -0.037, d1 +0.007. Lost; round 27 stays. Together with rounds 47
and 50 this is the third failed d1 training round: the 4B student looks capacity-limited on d1.
