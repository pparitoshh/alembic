#!/bin/bash
#SBATCH --job-name=dk-slx
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
# One model answers the 30 execution-eval tasks on a Leonardo GPU node (same prompt and settings as answer_task.sh on
# Curiosity); grade the output with grade.py on the login node.
# usage: REPO=<repo checkout> MODEL=<model dir> NAME=<label> OUT=<answers dir> sbatch --export=ALL answer_leonardo.sbatch
cd "${REPO:?}"
source research/rag65/env.sh; source slurm/common.sh
PORT=8301
start_vllm "${MODEL:?}" $PORT --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.85 --served-model-name "${NAME:?}"
mkdir -p "${OUT:?}"
"$VENV/bin/python" research/slurm_exec_eval/answer_tasks.py research/slurm_exec_eval/tasks.jsonl "http://127.0.0.1:$PORT/v1" "$NAME" "$OUT/$NAME.jsonl"
