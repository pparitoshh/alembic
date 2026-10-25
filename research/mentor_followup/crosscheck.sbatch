#!/bin/bash
#SBATCH --job-name=dk-mentor-real-pairs
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
mkdir -p "$RUN_DIR/mentor_followup"
"$VENV/bin/python" - <<'CHECK'
import hashlib,json,os
from pathlib import Path
repo=Path(os.environ['REPO'])
for name,sha in json.loads((Path(os.environ['RUN_DIR'])/'mentor_code_snapshot.json').read_text()).items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha,'Mentor code changed: '+name
CHECK
phase mentor_absolute_metrics
"$VENV/bin/python" research/mentor_followup/absolute_metrics.py "$CONFIG"
phase mentor_second_judge_calibration
PORT=$(job_port 6)
start_vllm google/gemma-4-26B-A4B-it "$PORT" --max-model-len 24576 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
distillkit calibrate -c "$CONFIG" --set run_dir="$RUN_DIR/mentor_followup" --set judge.base_url="http://127.0.0.1:$PORT/v1"
"$VENV/bin/python" - <<'PY'
import json,os
from pathlib import Path
r=json.loads((Path(os.environ['RUN_DIR'])/'mentor_followup/calibration_summary.json').read_text())
assert r['unparsed']==0 and r['accuracy']>=.8 and r['position_consistency']>=.8,'Gemma calibration failed'
PY
phase mentor_real_pair_agreement
"$VENV/bin/python" research/mentor_followup/real_pair_review.py "$CONFIG" "http://127.0.0.1:$PORT/v1"
stop_vllm
phase mentor_crosscheck_complete
