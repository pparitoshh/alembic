#!/bin/bash
#SBATCH --job-name=dk-generate
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=slurm/logs/%x-%j.out
# Teacher data: Qwen3-32B-AWQ served on this job's GPU, then generate (resumable) + verify + decontam.
# Killed at the wall time? Resubmit: generate continues where it stopped.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

PORT=$(job_port)
start_vllm "$TEACHER_MODEL" "$PORT" $(teacher_args)
distillkit generate -c "$CONFIG" --set run_dir="$RUN_DIR" --set teacher.base_url="http://127.0.0.1:$PORT/v1" --set teacher.model="$TEACHER_MODEL"
stop_vllm
distillkit verify -c "$CONFIG" --set run_dir="$RUN_DIR"
# verify only checks eval.file; decontam checks every data/eval/ file and the source holdouts.
# Training reads verified.jsonl, so it becomes the decontaminated set (the unfiltered copy is kept).
distillkit decontam -c "$CONFIG" --set run_dir="$RUN_DIR"
cp "$RUN_DIR/verified.jsonl" "$RUN_DIR/verified.pre_decontam.jsonl"
cp "$RUN_DIR/clean.jsonl" "$RUN_DIR/verified.jsonl"
