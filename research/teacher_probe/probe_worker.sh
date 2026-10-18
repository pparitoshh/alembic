#!/bin/bash
source "$REPO/research/rag65/env.sh"
source "$REPO/slurm/common.sh"  # start_vllm blocks until the server is healthy
cd "$REPO"
export PYTHONPATH=src:research/rag65
OUT=runs/teacher_probe/answers
P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
if [ "$SLURM_PROCID" = 0 ]; then
    start_vllm QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ $P --tensor-parallel-size 4 --max-model-len 16384 \
        --gpu-memory-utilization 0.92 --served-model-name t235
    "$VENV/bin/python" research/teacher_probe/answer_teacher.py t235 "http://127.0.0.1:$P/v1" t235 $OUT 0 rag,closed
else
    CUDA_VISIBLE_DEVICES=0,1 start_vllm Qwen/Qwen3-32B-AWQ $((P + 4)) --tensor-parallel-size 2 --max-model-len 16384 \
        --gpu-memory-utilization 0.90 --served-model-name t32
    CUDA_VISIBLE_DEVICES=2,3 start_vllm Qwen/Qwen3-32B-AWQ $((P + 8)) --tensor-parallel-size 2 --max-model-len 16384 \
        --gpu-memory-utilization 0.90 --served-model-name t32
    "$VENV/bin/python" research/teacher_probe/answer_teacher.py t32think "http://127.0.0.1:$((P + 4))/v1" t32 $OUT 1 rag &
    "$VENV/bin/python" research/teacher_probe/answer_teacher.py t32 "http://127.0.0.1:$((P + 8))/v1" t32 $OUT 0 rag &
    wait
fi
