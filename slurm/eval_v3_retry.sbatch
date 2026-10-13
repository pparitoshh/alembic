#!/bin/bash
#SBATCH --job-name=dk-eval-v3-retry
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=slurm/logs/%x-%j.out
# Reuse the 512 candidates. Original run files remain in their original directory.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_DIR=runs/eval_v3_repair_20261012
SET=(--set run_dir="$RUN_DIR" --set verify.grounding_reviews="$RUN_DIR/grounding_reviews.jsonl")
"$VENV/bin/python" -m distillkit.eval_v3 preflight --run-dir "$RUN_DIR"
PORT=$(job_port 1)
start_vllm "$CROSS_JUDGE_MODEL" "$PORT" $(judge_args "$CROSS_JUDGE_MODEL") \
    --max-model-len 24576 --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
# Maximum two format-only attempts per candidate, enforced persistently by the helper.
for attempt in 1 2; do
    "$VENV/bin/python" research/eval_v3/retry_support.py --run-dir "$RUN_DIR" --url "http://127.0.0.1:$PORT/v1"
    distillkit verify -c configs/gen_eval_v3.yaml "${SET[@]}"
done
"$VENV/bin/python" -m distillkit.eval_v3 review --run-dir "$RUN_DIR" \
    --review-url "http://127.0.0.1:$PORT/v1" --review-model "$CROSS_JUDGE_MODEL"
stop_vllm
"$VENV/bin/python" -m distillkit.eval_v3 release --run-dir "$RUN_DIR"
