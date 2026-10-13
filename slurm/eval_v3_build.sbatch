#!/bin/bash
#SBATCH --job-name=dk-eval-v3-build
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
# Evaluation candidates only. No training, gold copying, or historical-run writes.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG=configs/gen_eval_v3.yaml
RUN_DIR="${EVAL_BUILD_RUN:-runs/eval_v3_generation}"
SET=(--set run_dir="$RUN_DIR" --set verify.grounding_reviews="$RUN_DIR/grounding_reviews.jsonl")
"$VENV/bin/python" -m distillkit.eval_v3 preflight --config "$CONFIG" --run-dir "$RUN_DIR"
PORT=$(job_port)
start_vllm "$TEACHER_MODEL" "$PORT" $(teacher_args)
"$VENV/bin/python" -m distillkit.eval_v3 generate --config "$CONFIG" --run-dir "$RUN_DIR" \
    --teacher-url "http://127.0.0.1:$PORT/v1"
stop_vllm
PORT=$(job_port 1)
start_vllm "$CROSS_JUDGE_MODEL" "$PORT" $(judge_args "$CROSS_JUDGE_MODEL") \
    --max-model-len 24576 --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
distillkit review -c "$CONFIG" "${SET[@]}" --set judge.base_url="http://127.0.0.1:$PORT/v1" \
    --set judge.model="$CROSS_JUDGE_MODEL" --set judge.max_tokens=8192 \
    --set 'judge.extra_body={repetition_penalty: 1.05}'
distillkit verify -c "$CONFIG" "${SET[@]}"
"$VENV/bin/python" -m distillkit.eval_v3 review --config "$CONFIG" --run-dir "$RUN_DIR" \
    --review-url "http://127.0.0.1:$PORT/v1" --review-model "$CROSS_JUDGE_MODEL"
stop_vllm
"$VENV/bin/python" -m distillkit.eval_v3 release --config "$CONFIG" --run-dir "$RUN_DIR"
