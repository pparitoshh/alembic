#!/bin/bash
#SBATCH --job-name=dk-recovery-train
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%3
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/ai_factory_recovery.json"
export RUN_DIR="$REPO/runs/ai_factory_recovery_20261014"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
"$VENV/bin/python" - <<'SNAPSHOT'
import hashlib,json,os
from pathlib import Path
repo=Path(os.environ['REPO'])
for name,expected in json.loads((repo/'runs/ai_factory_recovery_20261014/recovery_code_snapshot.json').read_text()).items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected, 'Recovery input changed: '+name
SNAPSHOT
SEEDS=(42 43 44)
export SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:?Array task required}]}"
export SEED_DIR="$RUN_DIR/seed_$SEED"
"$VENV/bin/python" research/ai_factory_pilot/prepare_training.py "$CONFIG" "$SEED"
phase bf16_lora_pilot_training
distillkit train -c "$CONFIG" --set run_dir="$SEED_DIR" --set train.seed="$SEED"
phase frozen_evaluation_answers
for SLICE in v2 v3; do
    EVAL_DIR="$SEED_DIR/evaluation_$SLICE"
    mkdir -p "$EVAL_DIR"
    if [ ! -e "$EVAL_DIR/adapter" ]; then ln -s "$SEED_DIR/adapter" "$EVAL_DIR/adapter"; fi
    if [ "$SLICE" = v2 ]; then EVAL_FILE=data/eval/eval_all_v2.jsonl; else EVAL_FILE=data/eval/eval_v3.jsonl; fi
    distillkit answer -c "$CONFIG" --set run_dir="$EVAL_DIR" --set eval.file="$EVAL_FILE"
done
phase paired_order_evaluation
PORT=$(job_port 2)
start_vllm openai/gpt-oss-20b "$PORT" $(judge_args openai/gpt-oss-20b)
distillkit calibrate -c "$CONFIG" --set run_dir="$SEED_DIR" --set judge.base_url="http://127.0.0.1:$PORT/v1" \
    --set judge.model=openai/gpt-oss-20b --set 'judge.extra_body={reasoning_effort: medium}' \
    --set judge.max_tokens=4096
"$VENV/bin/python" - <<'PY'
import json, os
from pathlib import Path
r = json.loads((Path(os.environ['SEED_DIR']) / 'calibration_summary.json').read_text())
assert r['unparsed'] == 0 and r['accuracy'] >= 0.8 and r['position_consistency'] >= 0.8, \
    'Judge failed the existing synthetic calibration smoke gate; inspect before interpreting scores'
print('Synthetic calibration smoke gate passed. Real-pair/human calibration remains separate.')
PY
for SLICE in v2 v3; do
    if [ "$SLICE" = v2 ]; then EVAL_FILE=data/eval/eval_all_v2.jsonl; else EVAL_FILE=data/eval/eval_v3.jsonl; fi
    distillkit evaluate -c "$CONFIG" --set run_dir="$SEED_DIR/evaluation_$SLICE" --set eval.file="$EVAL_FILE" \
        --set judge.base_url="http://127.0.0.1:$PORT/v1" --set judge.model=openai/gpt-oss-20b \
        --set 'judge.extra_body={reasoning_effort: medium}' --set judge.max_tokens=4096
done
stop_vllm
"$VENV/bin/python" research/ai_factory_pilot/report.py "$SEED_DIR" "$SEED"
phase pilot_finished_not_released
