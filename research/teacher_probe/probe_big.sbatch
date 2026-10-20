#!/bin/bash
#SBATCH --job-name=dk-rag65-probe
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Bigger-teacher probe (Oct 20): TEACHER=t235 (Qwen3-235B-A22B-Instruct-2507 AWQ, TP4) or goss120 (gpt-oss-120b,
# reasoning medium) answers the eval sets with retrieval, like tnext. Checkpoint staged in node RAM. Resumable.
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH=src:research/rag65
P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
case "${TEACHER:?}" in
    t235) HUB=QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ; THINK=0 ;;
    goss120) HUB=openai/gpt-oss-120b; THINK=1 ;;
esac
SNAP=$(ls -d "$HF_HOME"/hub/models--${HUB/\//--}/snapshots/* | head -1)
STAGE=/dev/shm/probe-$SLURM_JOB_ID; mkdir -p $STAGE
for f in "$SNAP"/*; do cp -L "$f" $STAGE/ & done; wait
start_vllm $STAGE $P --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name $TEACHER
trap 'stop_vllm; rm -rf "$STAGE"' EXIT
"$VENV/bin/python" research/teacher_probe/answer_teacher.py $TEACHER "http://127.0.0.1:$P/v1" $TEACHER runs/teacher_probe4/answers $THINK rag
