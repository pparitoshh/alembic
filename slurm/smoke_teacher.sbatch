#!/bin/bash
#SBATCH --job-name=dk-smoke-teacher
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out
# Day-1 check (debug QOS, 30 min): the teacher serves on one A100, returns top-20 logprobs with token
# ids that decode with the student's tokenizer, emits parseable tool calls, and how fast it is.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

PORT=$(job_port)
start_vllm "$TEACHER_MODEL" "$PORT" $(teacher_args)
"$VENV/bin/python" slurm/smoke_teacher.py --base-url "http://127.0.0.1:$PORT/v1" --model "$TEACHER_MODEL" --student "$STUDENT_MODEL"
