#!/bin/bash
#SBATCH --job-name=dk-tool-recovery
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/tool_recovery.json"
export RUN_DIR="$REPO/runs/tool_recovery_20261014"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
"$VENV/bin/python" - <<'SNAPSHOT'
import hashlib,json,os
from pathlib import Path
repo=Path(os.environ['REPO'])
for name,expected in json.loads((repo/'runs/ai_factory_recovery_20261014/recovery_code_snapshot.json').read_text()).items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected, 'Recovery input changed: '+name
SNAPSHOT
phase corrected_tool_question_generation
PORT=$(job_port 9)
start_vllm Qwen/Qwen3-32B-AWQ "$PORT" $(teacher_args)
distillkit generate -c "$CONFIG" --set teacher.base_url="http://127.0.0.1:$PORT/v1"
stop_vllm
phase independent_tool_source_review
PORT=$(job_port 10)
start_vllm google/gemma-4-26B-A4B-it "$PORT" --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
distillkit review -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$PORT/v1"
stop_vllm
phase preserving_original_acceptance_and_combining
"$VENV/bin/python" research/tool_recovery/combine.py
export CONFIG="$REPO/configs/ai_factory_recovery.json"
export RUN_DIR="$REPO/runs/ai_factory_recovery_20261014"
distillkit verify -c "$CONFIG"
distillkit decontam -c "$CONFIG"
"$VENV/bin/python" research/ai_factory_pilot/finalize_data.py "$CONFIG"
phase recovery_data_gate_passed
