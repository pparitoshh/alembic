#!/bin/bash
#SBATCH --job-name=dk-eval-ckpt
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=slurm/logs/%x-%j.out
# Per seed: judge every kept checkpoint adapter (seed_<n>/adapters/step_<N>/) on the dataset_v1
# DEVELOPMENT slice, then log the results into the training run's MLflow run (`distillkit track`).
# Never point this at the final slice: choosing checkpoints on it would spend the final test.
# Only some steps or seeds:  STEPS="300 600 897" SEEDS=42 sbatch slurm/evaluate_checkpoints.sbatch
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
fan_out_seeds slurm/evaluate_checkpoints.sbatch   # one GPU task per seed; below runs once per seed

SEED_DIR="$RUN_DIR/seed_$(seed_of_task)"
DEV=(--set eval.file=data/eval/dataset_v1/development_normal.jsonl
     --set eval.tool_file=data/eval/dataset_v1/development_tools.jsonl
     --set eval.tool_protocol=tool-first-response-v2
     --set eval.tool_application_policy=hpc-tools-clarify-first-v1)
if [ -n "${STEPS:-}" ]; then DIRS=(); for s in $STEPS; do DIRS+=("$SEED_DIR/adapters/step_$s"); done
else mapfile -t DIRS < <(ls -d "$SEED_DIR"/adapters/step_* | sort -t_ -k2 -n); fi

for d in "${DIRS[@]}"; do   # GPU: base + student answers per checkpoint (cached, so a rerun skips done ones)
    distillkit answer -c "$CONFIG" --set run_dir="$d" --set student.model="$STUDENT_MODEL" "${DEV[@]}"
done
PORT=$(job_port 1)
start_vllm "$JUDGE_MODEL" "$PORT" $(judge_args "$JUDGE_MODEL")
for d in "${DIRS[@]}"; do
    distillkit evaluate -c "$CONFIG" --set run_dir="$d" --set student.model="$STUDENT_MODEL" "${DEV[@]}" \
        --set judge.base_url="http://127.0.0.1:$PORT/v1" --set judge.model="$JUDGE_MODEL"
done
stop_vllm
distillkit track -c "$CONFIG" --set run_dir="$SEED_DIR"
