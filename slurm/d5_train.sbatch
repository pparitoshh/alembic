#!/bin/bash
#SBATCH --job-name=dk-d5-train
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=slurm/logs/%x-%j.out
# D5 step 2, per seed: DPO with a new LoRA adapter on the start model (reference = start, adapter off).
# Writes runs/d5_dpo/seed_<n>/adapter. Submit: REPO=$WORK/alembic/wt-d5-dpo sbatch --dependency=afterok:<prepare> slurm/d5_train.sbatch
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src"
fan_out_seeds slurm/d5_train.sbatch

SEED=$(seed_of_task); D=runs/d5_dpo/seed_$SEED
"$VENV/bin/python" -m distillkit.dpo train --model "$D/start" --pairs "$D/pairs.jsonl" --out "$D" --seed "$SEED"
