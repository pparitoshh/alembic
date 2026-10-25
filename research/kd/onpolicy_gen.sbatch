#!/bin/bash
#SBATCH --job-name=dk-kd-opgen
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# On-policy distillation, step 1: the student (MODEL, a merged checkpoint) writes K (default 1) sampled answers (T=0.7, same
# system prompt as evaluation, no length note) per training question. One vLLM server per GPU, 8 workers on 2 nodes
# (keep --nodes=2: the row split depends on the worker count). Output: OUT/cands_<worker>.jsonl. Resumable.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --gpu-bind=single:1 --gpus-per-task=1 --output="$L/kd-opgen-%j-%t.out" --error="$L/kd-opgen-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
    start_vllm "${MODEL:?}" $P --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 --served-model-name student
    NO_LENGTH_NOTE=1 "$VENV/bin/python" research/bestofn/candidates.py $SLURM_PROCID $SLURM_NTASKS "http://127.0.0.1:$P/v1" student \
        "${DATA:?}" "${OUT:?}" ${K:-1}'
