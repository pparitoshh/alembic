#!/bin/bash
# One debug-QoS worker: shard = global task rank. PHASE=generate | review.
source "${REPO:?}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/ai_factory_ccby_v1_shard_${SLURM_PROCID}.json"
export RUN_DIR="$REPO/runs/ai_factory_ccby_v1/shard_${SLURM_PROCID}"
cd "$REPO"
# Distinct per task on a shared node: the job id keeps concurrent jobs apart.
PORT=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
case "${PHASE:?}" in
generate)
    phase teacher_generation
    start_vllm Qwen/Qwen3-32B-AWQ "$PORT" $(teacher_args)
    distillkit generate -c "$CONFIG" --set teacher.base_url="http://127.0.0.1:$PORT/v1"
    phase generation_complete ;;
review)
    phase independent_source_review
    start_vllm google/gemma-4-26B-A4B-it "$((PORT + 1))" --max-model-len 32768 --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
    distillkit review -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$((PORT + 1))/v1"
    stop_vllm
    phase deterministic_verification_and_decontamination
    distillkit verify -c "$CONFIG"
    distillkit decontam -c "$CONFIG"
    phase shard_complete ;;
esac
