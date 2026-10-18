#!/bin/bash
#SBATCH --job-name=dk-ai-factory-data
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
phase teacher_generation
PORT=$(job_port)
start_vllm Qwen/Qwen3-32B-AWQ "$PORT" $(teacher_args)
distillkit generate -c "$CONFIG" --set teacher.base_url="http://127.0.0.1:$PORT/v1"
stop_vllm
phase independent_source_review
PORT=$(job_port 1)
start_vllm google/gemma-4-26B-A4B-it "$PORT" --max-model-len 24576 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
distillkit review -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$PORT/v1"
stop_vllm
phase deterministic_verification_and_decontamination
distillkit verify -c "$CONFIG"
distillkit decontam -c "$CONFIG"
"$VENV/bin/python" research/ai_factory_pilot/finalize_data.py "$CONFIG"
phase verified_data_ready
