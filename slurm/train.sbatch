#!/bin/bash
#SBATCH --job-name=dk-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x-%A_%a.out
# QDoRA student, one seed per array task and per GPU (seeds from $SEEDS; --array must match its length).
# Each seed gets $RUN_DIR/seed_<n>/ sharing the verified data.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

SEED=$(seed_of_task)
SEED_DIR="$RUN_DIR/seed_$SEED"
mkdir -p "$SEED_DIR"
ln -sfn ../verified.jsonl "$SEED_DIR/verified.jsonl"
distillkit train -c "$CONFIG" --set run_dir="$SEED_DIR" --set train.seed="$SEED" --set student.model="$STUDENT_MODEL"
