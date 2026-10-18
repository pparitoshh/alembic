#!/bin/bash
#SBATCH --job-name=dk-student-with-rag
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/student_with_retrieval.json"
export RUN_DIR="$REPO/runs/student_with_retrieval_20261014"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
phase matched_student_rag_answers
"$VENV/bin/python" research/student_with_retrieval/experiment.py answer "$CONFIG"
phase matched_student_rag_calibration
PORT=$(job_port 14)
start_vllm openai/gpt-oss-20b "$PORT" $(judge_args openai/gpt-oss-20b)
distillkit calibrate -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$PORT/v1" \
  --set judge.model=openai/gpt-oss-20b --set 'judge.extra_body={reasoning_effort: medium}' --set judge.max_tokens=8192
"$VENV/bin/python" - <<'PY'
import json,os
from pathlib import Path
r=json.loads((Path(os.environ['RUN_DIR'])/'calibration_summary.json').read_text())
assert r['unparsed']==0 and r['accuracy']>=.8 and r['position_consistency']>=.8,'Judge calibration gate failed'
PY
phase matched_student_rag_judging
"$VENV/bin/python" research/student_with_retrieval/experiment.py judge "$CONFIG" "http://127.0.0.1:$PORT/v1"
stop_vllm
phase matched_student_rag_finished_not_released
