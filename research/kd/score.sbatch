#!/bin/bash
#SBATCH --job-name=dk-kd-score
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# sbatch --nodes=N: one Qwen3-Next TP4 server per node (checkpoint staged in node RAM) scoring 1/N of DATA. Resumable.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/kd-score-%j-%t.out" --error="$L/kd-score-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    SNAP=$(ls -d "$HF_HOME"/hub/models--cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit/snapshots/* | head -1)
    STAGE=/dev/shm/tnext-$SLURM_JOB_ID; mkdir -p $STAGE
    for f in "$SNAP"/*; do cp -L "$f" $STAGE/ & done; wait
    start_vllm $STAGE $P $(teacher_args) --tensor-parallel-size 4 --served-model-name tnext
    trap "stop_vllm; rm -rf $STAGE" EXIT
    "$VENV/bin/python" research/kd/score.py $SLURM_PROCID $SLURM_NTASKS "http://127.0.0.1:$P/v1" tnext "${DATA:?}" "${KD_DIR:?}"'
