#!/bin/bash
#SBATCH --job-name=dk-agent-smoke
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
# Agent smoke tests (research/agent_smoke/smoke.py): four models, one GPU each, served by vLLM with tool calling,
# each running the same 10 tasks against the real Slurm controller from this node. MODELS="name=path ..." (max 4).
source "${REPO:?}/research/rag65/env.sh"
OUT=${OUT:?}; ACCOUNT=${ACCOUNT:-EUHPC_D30_031}; PART=${PART:-lrd_all_serial}; INDEX=${INDEX:-runs/rag65/indexes}
export PYTHONPATH=src
command -v squeue > /dev/null && squeue --me -h > /dev/null || { echo "no Slurm client on $(hostname)"; exit 2; }
P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 )); i=0; ARGS=()
"$VENV/bin/python" research/agent_smoke/smoke.py "$OUT" "$ACCOUNT" "$PART" "$INDEX" &  # fixtures while the models load
FX=$!
# One server at a time: four parallel loads from Lustre took > 15 min (contention); one at a time fits the 30-min limit.
source "$REPO/slurm/common.sh"
for spec in ${MODELS:?}; do
    name=${spec%%=*}; path=${spec#*=}
    CUDA_VISIBLE_DEVICES=$i start_vllm "$path" $((P + i)) --served-model-name "$name" --dtype auto \
        --max-model-len 32768 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes
    ARGS+=("$name=http://127.0.0.1:$((P + i))/v1=$name"); i=$((i + 1))
done
wait $FX
PIDS=()
for a in "${ARGS[@]}"; do
    "$VENV/bin/python" research/agent_smoke/smoke.py "$OUT" "$ACCOUNT" "$PART" "$INDEX" "$a" > "$OUT/${a%%=*}.log" 2>&1 &
    PIDS+=($!)
done
wait "${PIDS[@]}"
cat "$OUT"/SUMMARY_*.json
