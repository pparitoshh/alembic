#!/bin/bash
#SBATCH --job-name=dk-rag65dpo
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Answer the eval sets with the DPO student (merged round-5 student + DPO adapter), closed-book and with retrieval,
# exactly like every student arm (same prompt, greedy, 768-token budget). Resumable.
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH=src:research/rag65
P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
start_vllm runs/dpo/merged_r5 $P --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules dpo=runs/dpo/out/adapter --served-model-name merged_r5
"$VENV/bin/python" research/teacher_probe/answer_teacher.py s43_dpo "http://127.0.0.1:$P/v1" dpo runs/dpo_eval/answers 0 rag,closed
