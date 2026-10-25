#!/bin/bash
#SBATCH --job-name=dk-rag65-probe
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Teacher probe: Qwen3-Next-80B-A3B-Instruct (AWQ 4-bit, TP4) answers the eval sets with retrieval, like t32.
# Resumable: resubmit to continue.
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH=src:research/rag65
P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
start_vllm cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit $P --tensor-parallel-size 4 --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --served-model-name tnext
"$VENV/bin/python" research/teacher_probe/answer_teacher.py tnext "http://127.0.0.1:$P/v1" tnext runs/teacher_probe2/answers 0 rag
