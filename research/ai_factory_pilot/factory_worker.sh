#!/bin/bash
# One debug-QoS worker: shard = SHARD_LIST[global task rank] (REVIEW_TP=2: two shards per two-GPU task).
# PHASE=generate | review. Both resume.
source "${REPO:?}/research/ai_factory_pilot/environment.sh"
read -r -a LIST <<< "${SHARD_LIST:?}"
BATCH=${BATCH:-ai_factory_all_v1}
cd "$REPO"
PORT=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
if [ "$PHASE" = review ] && [ "${REVIEW_TP:-1}" = 2 ]; then
    # Two GPUs per task, two shards per task (factory_dbg.sbatch). Gemma's bf16 weights fill ~52 of 64 GB, so a
    # one-GPU server holds ~15 concurrent reviews; split over two GPUs it has ~9x the KV cache and serves both
    # shards' requests at once. Same model, prompts and checks.
    MINE=()
    for i in $(( 2 * SLURM_PROCID )) $(( 2 * SLURM_PROCID + 1 )); do (( i < ${#LIST[@]} )) && MINE+=("${LIST[$i]}"); done
    (( ${#MINE[@]} )) || exit 0
    export CUDA_VISIBLE_DEVICES=$(( 2 * SLURM_LOCALID )),$(( 2 * SLURM_LOCALID + 1 ))
    start_vllm google/gemma-4-26B-A4B-it "$((PORT + 1))" --tensor-parallel-size 2 --max-model-len 32768 \
        --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0}' \
        --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
    REVIEWS=()
    for K in "${MINE[@]}"; do
        ( export CONFIG="$REPO/configs/${BATCH}_shard_${K}.json" RUN_DIR="$REPO/runs/$BATCH/shard_${K}"
          phase independent_source_review
          # 64 per shard, 128 per server: at 32 per shard the two-GPU KV cache sat at ~39% (Oct 18 measurement).
          distillkit review -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$((PORT + 1))/v1" --set judge.concurrency=64 ) &
        REVIEWS+=($!)
    done
    # Wait for the reviews only: a bare `wait` also waits for the vLLM server and never returns.
    wait "${REVIEWS[@]}" || true
    stop_vllm
    for K in "${MINE[@]}"; do
        export CONFIG="$REPO/configs/${BATCH}_shard_${K}.json" RUN_DIR="$REPO/runs/$BATCH/shard_${K}"
        phase deterministic_verification_and_decontamination
        distillkit verify -c "$CONFIG"
        distillkit decontam -c "$CONFIG"
        touch "$RUN_DIR/REVIEW_V2_DONE"
        phase shard_complete
    done
    exit 0
fi
if [ "$PHASE" = generate ] && [ "${GEN_TP:-1}" = 4 ]; then
    # Qwen3-Next teacher (user decision, Oct 19: it beat Qwen3-32B by 8.9 points on v3 in one judge run): one
    # TP4 server per node, one task per node, four shards generating against it at once (factory_dbg.sbatch).
    MINE=()
    for i in $(( 4 * SLURM_PROCID )) $(( 4 * SLURM_PROCID + 1 )) $(( 4 * SLURM_PROCID + 2 )) $(( 4 * SLURM_PROCID + 3 )); do
        (( i < ${#LIST[@]} )) && MINE+=("${LIST[$i]}")
    done
    (( ${#MINE[@]} )) || exit 0
    TNEXT=cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit
    # Stage the 46 GB checkpoint in node RAM first: ten parallel sequential copies take well under a minute, while
    # vLLM's four TP ranks prefetching from Lustre with 8 threads each took 764 s (Oct 19, job 61459978).
    SNAP=$(ls -d "$HF_HOME"/hub/models--cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit/snapshots/* | head -1)
    STAGE=/dev/shm/tnext-$SLURM_JOB_ID
    mkdir -p $STAGE && trap 'rm -rf "$STAGE"' EXIT
    t0=$SECONDS
    for f in "$SNAP"/*; do cp -L "$f" $STAGE/ & done
    wait
    echo "[job] staged $(du -sh $STAGE | cut -f1) in node RAM in $(( SECONDS - t0 )) s"
    # Same flags as the 32B teacher (logprobs, hermes tool calls); served under the hub name the configs use.
    start_vllm $STAGE "$PORT" $(teacher_args) --tensor-parallel-size 4 --served-model-name $TNEXT
    trap 'stop_vllm; rm -rf "$STAGE"' EXIT  # start_vllm replaced the trap: free the node RAM too
    GENS=()
    for K in "${MINE[@]}"; do
        ( export CONFIG="$REPO/configs/${BATCH}_shard_${K}.json" RUN_DIR="$REPO/runs/$BATCH/shard_${K}"
          phase teacher_generation
          distillkit generate -c "$CONFIG" --set teacher.base_url="http://127.0.0.1:$PORT/v1" --set teacher.model=$TNEXT \
              && touch "$RUN_DIR/GENERATE_DONE" && phase generation_complete ) &
        GENS+=($!)
    done
    wait "${GENS[@]}" || true
    exit 0
fi
(( SLURM_PROCID < ${#LIST[@]} )) || exit 0
K=${LIST[$SLURM_PROCID]}
export CONFIG="$REPO/configs/${BATCH}_shard_${K}.json"
export RUN_DIR="$REPO/runs/$BATCH/shard_${K}"
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
