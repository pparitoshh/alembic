#!/bin/bash
#SBATCH --job-name=dk-t120-regen
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Teacher upgrade: one gpt-oss-120b server (TP4) per node runs research/t120/regen.py on 1/N of DATA into OUT.
# Keep --nodes fixed across attempts (the row split depends on the worker count). Resumable.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/t120-regen-%j-%t.out" --error="$L/t120-regen-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    start_vllm openai/gpt-oss-120b $P --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name t120
    "$VENV/bin/python" research/t120/regen.py $SLURM_PROCID $SLURM_NTASKS "http://127.0.0.1:$P/v1" t120 "${DATA:?}" "${OUT:?}"'
