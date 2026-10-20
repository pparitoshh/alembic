#!/bin/bash
# Best-of-n worker. PHASE=cands: TEACHER=tnext (Qwen3-Next-80B-A3B AWQ, TP4, one task per node) or t32
# (Qwen3-32B-AWQ, one GPU per task). PHASE=select: Gemma split over two GPUs, two tasks per node.
# Shard = SHARD_LIST[task]; each shard writes its own files and a done-marker. Resumable.
source "${REPO:?}/research/ai_factory_pilot/environment.sh"
read -r -a LIST <<< "${SHARD_LIST:?}"
(( SLURM_PROCID < ${#LIST[@]} )) || exit 0
K=${LIST[$SLURM_PROCID]}
cd "$REPO"
export PYTHONPATH="$REPO/src"
B=${BON_DIR:-runs/bestofn}  # a later run (e.g. runs/bestofn2) passes BON_DIR
PORT=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
case "${PHASE:?}" in
cands)
    if [ "${TEACHER:?}" = tnext ]; then
        start_vllm cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit "$PORT" --tensor-parallel-size 4 --max-model-len 16384 \
            --gpu-memory-utilization 0.90 --served-model-name tnext
    else
        start_vllm Qwen/Qwen3-32B-AWQ "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name t32
    fi
    "$VENV/bin/python" research/bestofn/candidates.py "$K" "$NSHARDS" "http://127.0.0.1:$PORT/v1" "$TEACHER" "$B/data.jsonl" "$B/cands" 2  # best-of-3 added only 0.7-2.4 points over one Qwen3-Next answer
    touch "$B/cands/DONE_$K" ;;
select)
    export CUDA_VISIBLE_DEVICES=$(( 2 * SLURM_LOCALID )),$(( 2 * SLURM_LOCALID + 1 ))
    start_vllm google/gemma-4-26B-A4B-it "$((PORT + 1))" --tensor-parallel-size 2 --max-model-len 32768 \
        --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0}' \
        --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' --served-model-name gemma
    "$VENV/bin/python" research/bestofn/select.py "$K" "$NSHARDS" "http://127.0.0.1:$((PORT + 1))/v1" gemma "$B/data.jsonl" "$B/cands" "$B/select"
    touch "$B/select/DONE_$K" ;;
esac
