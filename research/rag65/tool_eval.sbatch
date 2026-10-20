#!/bin/bash
#SBATCH --job-name=dk-rag65-tooleval
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# G2 tool-call validity for base Qwen3-4B and the adapters of RUN (default rag65v4). One GPU.
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
RUN=${RUN:-rag65v4}
T=runs/$RUN/train
PORT=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
MODS=$(for s in 42 43 44; do for v in "" _x05 _x02; do echo "s$s$v=$T/seed$s/adapter$v"; done; done)
start_vllm Qwen/Qwen3-4B-Instruct-2507 "$PORT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
    --enable-lora --max-lora-rank 16 --max-loras 9 --lora-modules $MODS
"$VENV/bin/python" research/rag65/tool_eval.py "http://127.0.0.1:$PORT/v1" "runs/$RUN/TOOL_EVAL.json" \
    Qwen/Qwen3-4B-Instruct-2507 $(for s in 42 43 44; do for v in "" _x05 _x02; do echo -n "s$s$v "; done; done)
