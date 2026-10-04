#!/bin/bash
#SBATCH --job-name=dk-train-fsdp
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
# One QDoRA run over 2 GPUs with FSDP (configs/accelerate/fsdp.yaml, num_processes must match --gres).
# Untested on Leonardo: try on boost_qos_dbg first. For seed variance, prefer train.sbatch (1 GPU per seed).
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

SEED="${SEED:-42}"
SEED_DIR="$RUN_DIR/fsdp_seed_$SEED"
mkdir -p "$SEED_DIR"
ln -sfn ../verified.jsonl "$SEED_DIR/verified.jsonl"
"$VENV/bin/accelerate" launch --config_file configs/accelerate/fsdp.yaml -m distillkit.cli train \
    -c "$CONFIG" --set run_dir="$SEED_DIR" --set train.seed="$SEED" --set student.model="$STUDENT_MODEL"
