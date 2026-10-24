#!/bin/bash
#SBATCH --job-name=dk-op235-score
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# research/kd/score.py with Qwen3-235B-A22B-Instruct-2507 (AWQ, TP4 per node, staged in node RAM) as the teacher:
# the strongest model on our leaderboard (majority 0.875) and the same vocabulary and chat template as Qwen3-4B.
# sbatch --nodes=N (keep N fixed across attempts: the row split depends on it). Resumable.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/op235-score-%j-%t.out" --error="$L/op235-score-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    SNAP=$(ls -d "$HF_HOME"/hub/models--QuantTrio--Qwen3-235B-A22B-Instruct-2507-AWQ/snapshots/* | head -1)
    STAGE=/dev/shm/q235-$SLURM_JOB_ID; mkdir -p $STAGE
    for f in "$SNAP"/*; do cp -rL "$f" $STAGE/ & done; wait
    trap "stop_vllm; rm -rf $STAGE" EXIT
    start_vllm $STAGE $P --max-logprobs 20 --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.92 --served-model-name qwen3-235b
    "$VENV/bin/python" research/kd/score.py $SLURM_PROCID $SLURM_NTASKS "http://127.0.0.1:$P/v1" qwen3-235b "${DATA:?}" "${KD_DIR:?}"'
