#!/bin/bash
# One debug-QoS worker: shard = SHARD_LIST[global task rank]. PHASE=generate | review. Both resume.
source "${REPO:?}/research/ai_factory_pilot/environment.sh"
read -r -a LIST <<< "${SHARD_LIST:?}"
(( SLURM_PROCID < ${#LIST[@]} )) || exit 0
K=${LIST[$SLURM_PROCID]}
BATCH=${BATCH:-ai_factory_all_v1}
export CONFIG="$REPO/configs/${BATCH}_shard_${K}.json"
export RUN_DIR="$REPO/runs/$BATCH/shard_${K}"
cd "$REPO"
# Distinct per task on a shared node: the job id keeps concurrent jobs apart.
PORT=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
case "${PHASE:?}" in
generate)
    phase teacher_generation
    start_vllm Qwen/Qwen3-32B-AWQ "$PORT" $(teacher_args)
    distillkit generate -c "$CONFIG" --set teacher.base_url="http://127.0.0.1:$PORT/v1"
    touch "$RUN_DIR/GENERATE_DONE"
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
    touch "$RUN_DIR/REVIEW_V2_DONE"
    phase shard_complete ;;
esac
