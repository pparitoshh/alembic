#!/bin/bash
#SBATCH --job-name=dk-rag65-probe
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Third judge (Qwen3-235B-A22B AWQ, TP4 per node) on RUN's answers -> runs/$RUN/judged_qwen; sbatch --nodes=N.
# Same prompt, both orders and scoring as gpt-oss-20b and Gemma; a third model family so research/rag65/majority.py can
# take a 2-of-3 vote. gpt-oss-120b is not used as a judge: it is the round-19 teacher. Checkpoint staged in node RAM.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/judge-qwen-%j-%t.out" --error="$L/judge-qwen-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    SNAP=$(ls -d "$HF_HOME"/hub/models--QuantTrio--Qwen3-235B-A22B-Instruct-2507-AWQ/snapshots/* | head -1)
    STAGE=/dev/shm/q235-$SLURM_JOB_ID; mkdir -p $STAGE
    for f in "$SNAP"/*; do cp -rL "$f" $STAGE/ & done; wait
    trap "rm -rf $STAGE" EXIT
    start_vllm $STAGE $P --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.92 --served-model-name qwen3-235b
    JUDGE_MODEL=qwen3-235b "$VENV/bin/python" research/rag65/judge.py $SLURM_PROCID $SLURM_NTASKS \
        "http://127.0.0.1:$P/v1" "runs/${RUN:?}/answers" "runs/$RUN/judged_qwen"'
