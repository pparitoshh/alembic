#!/bin/bash
#SBATCH --job-name=dk-evaluate
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x-%A_%a.out
# Per seed, on one GPU: 1) base + student answers (cached in seed_<n>/eval_answers.json),
# 2) the judge served on the same GPU scores them.
# Cross-check judge on the same answers:  JUDGE=cross sbatch slurm/evaluate.sbatch
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

SEED=$(seed_of_task)
SEED_DIR="$RUN_DIR/seed_$SEED"
if [ "${JUDGE:-main}" = "cross" ]; then MODEL="$CROSS_JUDGE_MODEL"; TAG="gemma4"; else MODEL="$JUDGE_MODEL"; TAG=""; fi

distillkit answer -c "$CONFIG" --set run_dir="$SEED_DIR" --set student.model="$STUDENT_MODEL"
PORT=$(job_port 1)
start_vllm "$MODEL" "$PORT" $(judge_args "$MODEL")
distillkit evaluate -c "$CONFIG" --set run_dir="$SEED_DIR" --set student.model="$STUDENT_MODEL" \
    --set judge.base_url="http://127.0.0.1:$PORT/v1" --set judge.model="$MODEL" --set eval.tag="$TAG"
