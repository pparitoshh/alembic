#!/bin/bash
#SBATCH --job-name=dk-tooleval
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Tool-call validity for up to 4 models, one per GPU: vLLM with hermes tool parsing serves MODEL (and an optional
# LoRA adapter), research/tools38/tool_eval.py scores the held-out workflows and eval_tools.jsonl.
# MODELS="name=model_path[:adapter_dir] ..."  OUT=results dir  TEST=tool_test.jsonl
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --gpu-bind=single:1 --gpus-per-task=1 --output="$L/tooleval-%j-%t.out" --error="$L/tooleval-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src
    M=(${MODELS:?}); (( SLURM_PROCID < ${#M[@]} )) || exit 0
    spec=${M[$SLURM_PROCID]}; name=${spec%%=*}; rest=${spec#*=}; model=${rest%%:*}; lora=""; [[ "$rest" == *:* ]] && lora=${rest#*:}
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
    extra=(--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --gpu-memory-utilization 0.85)
    if [ -n "$lora" ]; then extra+=(--enable-lora --max-lora-rank 64 --lora-modules $name=$lora); served=$name; else extra+=(--served-model-name $name); served=$name; fi
    start_vllm "$model" $P "${extra[@]}"
    mkdir -p "${OUT:?}"
    "$VENV/bin/python" research/tools38/tool_eval.py "http://127.0.0.1:$P/v1" $served "${TEST:?}" data/eval/eval_tools.jsonl "$OUT/$name.json"'
