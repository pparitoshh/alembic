#!/bin/bash
#SBATCH --job-name=dk-ccby-data
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --array=0-7
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# One shard per array task: teacher generation -> independent Gemma source review -> deterministic checks.
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/ai_factory_ccby_v1_shard_${SLURM_ARRAY_TASK_ID}.json"
export RUN_DIR="$REPO/runs/ai_factory_ccby_v1/shard_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
phase teacher_generation
PORT=$(job_port)
start_vllm Qwen/Qwen3-32B-AWQ "$PORT" $(teacher_args)
distillkit generate -c "$CONFIG" --set teacher.base_url="http://127.0.0.1:$PORT/v1"
stop_vllm
phase independent_source_review
PORT=$(job_port 1)
start_vllm google/gemma-4-26B-A4B-it "$PORT" --max-model-len 32768 --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
distillkit review -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$PORT/v1"
stop_vllm
phase deterministic_verification_and_decontamination
distillkit verify -c "$CONFIG"
distillkit decontam -c "$CONFIG"
phase shard_complete
