#!/bin/bash
#SBATCH --job-name=dk-rag65-probe
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Cross-check judge (Gemma 4 26B, TP4 per node) on RUN's answers -> runs/$RUN/judged_gemma; sbatch --nodes=N.
# Same prompt, both orders and scoring as the gpt-oss-20b judge; a second family rules out judge self-preference.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/judge-gemma-%j-%t.out" --error="$L/judge-gemma-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    start_vllm google/gemma-4-26B-A4B-it $P --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt "{\"image\":0}" --served-model-name google/gemma-4-26B-A4B-it
    JUDGE_MODEL=google/gemma-4-26B-A4B-it "$VENV/bin/python" research/rag65/judge.py $SLURM_PROCID $SLURM_NTASKS \
        "http://127.0.0.1:$P/v1" "runs/${RUN:?}/answers" "runs/$RUN/judged_gemma"'
