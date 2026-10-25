#!/bin/bash
#SBATCH --job-name=dk-bot23-select
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Round 23 selection: Qwen3-Next-80B (not an evaluation judge; TP4 per node) picks the better of the verified answer
# and Qwen3-235B's answer for each training question (research/bestofn/select.py). sbatch --nodes=N; keep N fixed.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/bot23-select-%j-%t.out" --error="$L/bot23-select-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    start_vllm cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit $P --tensor-parallel-size 4 --max-model-len 32768 \
        --gpu-memory-utilization 0.90 --structured-outputs-config "{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}" --served-model-name tnext
    "$VENV/bin/python" research/bestofn/select.py $SLURM_PROCID $SLURM_NTASKS "http://127.0.0.1:$P/v1" tnext "${DATA:?}" "${CANDS:?}" "${SEL:?}"'
