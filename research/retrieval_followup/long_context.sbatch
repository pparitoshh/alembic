#!/bin/bash
#SBATCH --job-name=dk-rag-2048
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/retrieval_2048.json"
export RUN_DIR="$REPO/runs/retrieval_2048_20261014"
mkdir -p "$RUN_DIR/retrieval"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
phase retrieval_2048_answers
PORT=$(job_port 7)
start_vllm Qwen/Qwen3-4B-Instruct-2507 "$PORT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85
"$VENV/bin/python" research/ai_factory_pilot/retrieval_benchmark.py answer "$CONFIG" "http://127.0.0.1:$PORT/v1"
stop_vllm
phase retrieval_2048_judging
PORT=$(job_port 8)
start_vllm openai/gpt-oss-20b "$PORT" $(judge_args openai/gpt-oss-20b)
distillkit calibrate -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$PORT/v1" \
    --set judge.model=openai/gpt-oss-20b --set 'judge.extra_body={reasoning_effort: medium}' --set judge.max_tokens=4096
"$VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['RUN_DIR']) / 'calibration_summary.json').read_text())
assert r['unparsed'] == 0 and r['accuracy'] >= .8 and r['position_consistency'] >= .8, 'Calibration gate failed'
PY
if ! "$VENV/bin/python" research/ai_factory_pilot/retrieval_benchmark.py judge "$CONFIG" "http://127.0.0.1:$PORT/v1"; then
    test -s "$RUN_DIR/retrieval/BASE_RAG_RESULT.json"
fi
"$VENV/bin/python" research/retrieval_followup/repair_judgments.py --config "$CONFIG" \
    --endpoint "http://127.0.0.1:$PORT/v1" --source "$RUN_DIR/retrieval" --output "$RUN_DIR/retrieval_repaired"
stop_vllm
phase retrieval_2048_complete
