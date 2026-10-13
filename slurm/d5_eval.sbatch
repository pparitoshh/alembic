#!/bin/bash
#SBATCH --job-name=dk-d5-eval
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=slurm/logs/%x-%j.out
# D5 step 3, per seed: merge the DPO adapter into the start model, then greedy v2/v3 answers to
# runs/scaled_eval/b7s<n>d5_answers.json. Judge after it with $WORK/alembic/jobs/judge_dirs.sbatch
# (it builds judge_b7s<n>d5_t1.0_<v2|v3> against the cached base answers).
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src"
fan_out_seeds slurm/d5_eval.sbatch

SEED=$(seed_of_task); D=runs/d5_dpo/seed_$SEED; PY="$VENV/bin/python"
[ -f "$D/final/config.json" ] || "$PY" -m distillkit.merge "$D/start" "$D/adapter" "$D/final"
"$PY" -m distillkit.dpo answers --model "$D/final" --out runs/scaled_eval/b7s${SEED}d5_answers.json
