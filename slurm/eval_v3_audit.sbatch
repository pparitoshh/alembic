#!/bin/bash
#SBATCH --job-name=dk-eval-v3-audit
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=slurm/logs/%x-%j.out
# No GPU needed. Audit historical data and create separate future-merge inputs.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
: "${TRAIN_RUNS:?Set TRAIN_RUNS to the existing training runs directory}"
"$VENV/bin/python" -m distillkit.eval_v3 check-release
INPUTS=(--input "$TRAIN_RUNS/hpc10k/clean.jsonl")
for run in hpc10k_f1 hpc10k_f2 hpc10k_f3 hpc10k_f4 hpc10k_f5 hpc10k_b1 hpc10k_b2 hpc10k_b3; do
    test -s "$TRAIN_RUNS/$run/clean.jsonl"
    INPUTS+=(--input "$TRAIN_RUNS/$run/clean.jsonl")
done
# B6 is audited as historical exposure, never relabelled as clean training.
INPUTS+=(--input "$TRAIN_RUNS/b6_hpc10k/verified.jsonl")
"$VENV/bin/python" -m distillkit.eval_v3 audit "${INPUTS[@]}" --output runs/eval_v3_audit
