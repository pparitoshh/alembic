# Round 50: train the "better of two" answer into the model (Oct 25)

The GPU/MPI analysis (`research/gpu_mpi/`) shows round 27 could gain about +0.10 on d1 by picking, per question, the
better of its answer with excerpts and its closed-book answer. Round 50 trains that choice in:

1. `r50_build.py`: 2,000 GPU/MPI/performance and software-environment questions from the verified banks (held-out
   families and near-eval questions excluded), each with the app's excerpts (variant e) and its verified reference.
2. `r50_gen_two.py`, `r50_gen.sbatch`: round 27 answers each one with and without the excerpts.
3. Qwen3-235B judges the two answers with the reference (both orders), through `research/rag65/judge.py` with `EVAL_DIR`.
4. `r50_train_data.py`: the target is the better answer, always with the excerpts in the prompt, plus replay rows.
5. `r50.sh`: the whole chain on Leonardo, then training (3 seeds, x0.5 soup into round 27) and the comparison with
   round 27 on all three sets.

Unlike round 47 (token-level correction of the with-excerpts answer, which lost), the target here can be the model's
own closed-book answer, so it learns to fall back on its own knowledge when the excerpts do not help.

Result (Oct 25, 3-judge majority vs round 27): v2 -0.034, v3 +0.011, d1 -0.012. Lost: the model did not learn when to
ignore the excerpts. See `research/round51/` for the next attempt.
