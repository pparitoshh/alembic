#!/bin/bash
#SBATCH --job-name=dk-rag-second-judge
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
export CONFIG="$REPO/configs/retrieval_crosscheck.json"
export RUN_DIR="$REPO/runs/retrieval_crosscheck_20261014"
mkdir -p "$RUN_DIR"
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
"$VENV/bin/python" - <<'PY'
import hashlib,json,os
from pathlib import Path
repo=Path(os.environ['REPO'])
for path,sha in json.loads((Path(os.environ['RUN_DIR'])/'input_snapshot.json').read_text()).items():
    assert hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha, 'RAG crosscheck input changed: '+path
PY
phase rag_second_judge_calibration
PORT=$(job_port 11)
start_vllm google/gemma-4-26B-A4B-it "$PORT" --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
distillkit calibrate -c "$CONFIG" --set judge.base_url="http://127.0.0.1:$PORT/v1"
"$VENV/bin/python" - <<'PY'
import json,os
from pathlib import Path
r=json.loads((Path(os.environ['RUN_DIR'])/'calibration_summary.json').read_text())
assert r['unparsed']==0 and r['accuracy']>=.8 and r['position_consistency']>=.8, 'Second-judge calibration gate failed'
PY
phase rag_actual_pair_crosscheck
"$VENV/bin/python" research/retrieval_crosscheck/review.py "$CONFIG" "http://127.0.0.1:$PORT/v1"
stop_vllm
phase rag_second_judge_complete
