#!/bin/bash
#SBATCH --job-name=dk-calibrate
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
# Judge trust check on known-label pairs (eval.calibration_file), before using a judge for results.
#   sbatch slurm/calibrate.sbatch               # gpt-oss-20b
#   JUDGE=cross sbatch slurm/calibrate.sbatch   # Gemma 4 26B-A4B
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

if [ "${JUDGE:-main}" = "cross" ]; then MODEL="$CROSS_JUDGE_MODEL"; TAG="gemma4"; else MODEL="$JUDGE_MODEL"; TAG="gpt-oss"; fi
PORT=$(job_port 1)
start_vllm "$MODEL" "$PORT" $(judge_args "$MODEL")
distillkit calibrate -c "$CONFIG" --set run_dir="$RUN_DIR" \
    --set judge.base_url="http://127.0.0.1:$PORT/v1" --set judge.model="$MODEL" --set eval.tag="$TAG"
