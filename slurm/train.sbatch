#!/bin/bash
#SBATCH --job-name=dk-train
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=slurm/logs/%x-%j.out
# Student training: one whole node, one GPU task per seed in $SEEDS (at most 4 seeds per node).
# Each seed gets $RUN_DIR/seed_<n>/ sharing the verified data.
# Killed at the wall time? Resubmit the job: each seed resumes from its newest checkpoint (train.save_steps).
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
fan_out_seeds slurm/train.sbatch   # one GPU task per seed; below runs once per seed

SEED=$(seed_of_task)
SEED_DIR="$RUN_DIR/seed_$SEED"
mkdir -p "$SEED_DIR"
ln -sfn ../verified.jsonl "$SEED_DIR/verified.jsonl"
distillkit train -c "$CONFIG" --set run_dir="$SEED_DIR" --set train.seed="$SEED" --set student.model="$STUDENT_MODEL"
