#!/bin/bash
# STAGE=regen (N workers, teacher), answer (4 workers: seeds 42/43/44 + base) or judge (N workers, gpt-oss-20b).
# RUN names the experiment directory under runs/ (default rag65). BASE_MODEL is the student's base for the answer
# stage (default Qwen3-4B-Instruct-2507; the configs' student.model must match).
source "$REPO/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
W=$SLURM_PROCID
PORT=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
RUN=${RUN:-rag65}
T=runs/$RUN/train
case "${STAGE:?}" in
regen)
    start_vllm Qwen/Qwen3-32B-AWQ "$PORT" $(teacher_args)
    "$VENV/bin/python" research/rag65/regen.py "$W" "$SLURM_NTASKS" "http://127.0.0.1:$PORT/v1" runs/rag65/raft/verified.jsonl "runs/$RUN/regen" ;;
answer)
    # One node: 4 workers (seeds 42/43/44 + base), both modes. Two nodes (8 tasks): node 1 answers closed-book,
    # node 2 with retrieval, so evaluation answers take half as long.
    if [ "${SLURM_NTASKS:-4}" = 8 ]; then
        export ANSWER_MODES=$( (( W < 4 )) && echo closed || echo rag )
    fi
    W=$(( W % 4 ))
    if (( W < 3 )); then
        S=(42 43 44); s=${S[$W]}
        # Seed soups (rank 48) live in one seed's directory only; each worker serves the variants it has.
        # `if`, not `[ ] &&`: under set -e a false test as the loop's last command fails the substitution.
        VARIANTS=$(IFS=,; for v in ${VARIANTS:-,_x05,_x02,_soup,_soup05}; do if [ -d "$T/seed$s/adapter$v" ]; then echo -n "$v,"; fi; done)
        export VARIANTS=${VARIANTS%,}
        start_vllm "${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}" "$PORT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
            --enable-lora --max-lora-rank 64 --max-loras 5 \
            --lora-modules $(IFS=,; for v in $VARIANTS; do echo "s$s$v=$T/seed$s/adapter$v"; done)
    else
        # BASE_SERVED (opt-in): name a merged base model as answer.py expects (round 27 soup answered as a plain model).
        start_vllm "${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}" "$PORT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 ${BASE_SERVED:+--served-model-name $BASE_SERVED}
    fi
    "$VENV/bin/python" research/rag65/answer.py "$W" "http://127.0.0.1:$PORT/v1" "runs/$RUN/answers" ;;
judge)
    start_vllm openai/gpt-oss-20b "$PORT" $(judge_args openai/gpt-oss-20b)
    if (( W == 0 )); then
        "$VENV/bin/python" -m distillkit.cli calibrate -c configs/rag65_seed42.json --set judge.base_url="http://127.0.0.1:$PORT/v1" \
            --set judge.model=openai/gpt-oss-20b --set 'judge.extra_body={reasoning_effort: medium}' --set judge.max_tokens=8192 || true
    fi
    "$VENV/bin/python" research/rag65/judge.py "$W" "$SLURM_NTASKS" "http://127.0.0.1:$PORT/v1" "runs/$RUN/answers" "runs/$RUN/judged" ;;
esac
