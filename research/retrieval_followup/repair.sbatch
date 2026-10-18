#!/bin/bash
#SBATCH --job-name=dk-rag-judge-repair
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
PORT=$(job_port 6)
phase retrieval_judge_repair
start_vllm openai/gpt-oss-20b "$PORT" $(judge_args openai/gpt-oss-20b)
"$VENV/bin/python" research/retrieval_followup/repair_judgments.py --config "$CONFIG" \
    --endpoint "http://127.0.0.1:$PORT/v1" --source "$RUN_DIR/retrieval" --output "$RUN_DIR/retrieval_repaired"
stop_vllm
phase retrieval_judge_repair_complete
