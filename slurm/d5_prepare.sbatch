#!/bin/bash
#SBATCH --job-name=dk-d5-prepare
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=slurm/logs/%x-%j.out
# D5 step 1 (docs/plan/STUDENT_DISTILL.md), per seed: start model = base + 0.2 * (B7 - base) in bf16,
# then vLLM serves it and mines DPO pairs from the good-quality subset. Writes only runs/d5_dpo/seed_<n>/.
# Submit from the worktree: REPO=$WORK/alembic/wt-d5-dpo sbatch slurm/d5_prepare.sbatch
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src"   # this worktree's distillkit, not the shared checkout's
fan_out_seeds slurm/d5_prepare.sbatch

SEED=$(seed_of_task); D=runs/d5_dpo/seed_$SEED; mkdir -p "$D"
PY="$VENV/bin/python"
[ -f "$D/start/config.json" ] || "$PY" -m distillkit.merge "$STUDENT_MODEL" runs/b7_hpc10k/seed_$SEED/adapter "$D/start" --scale ${SCALE:-0.2}
[ -f "$D/pairs.jsonl" ] && { echo "pairs exist: $D/pairs.jsonl"; exit 0; }
PORT=$(job_port 1)
start_vllm "$D/start" "$PORT" --max-model-len 4096 --gpu-memory-utilization 0.90 --seed "$SEED"
"$PY" -m distillkit.dpo mine --url "http://127.0.0.1:$PORT/v1" --model "$D/start" --train runs/b7_good/verified.jsonl --out "$D/pairs.jsonl"
stop_vllm
