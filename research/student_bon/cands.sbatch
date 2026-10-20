#!/bin/bash
#SBATCH --job-name=dk-rag65-sbon
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Student best-of-n at answer time: one task per seed serves the student adapter (RUN, VARIANT) and samples 3
# answers per eval question (same prompt as evaluation), or K answers into CDIR. Output: runs/$OUT/s<seed>/${CDIR:-cands}.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --gpu-bind=single:1 --gpus-per-task=1 --output="$L/sbon-cands-%j-%t.out" --error="$L/sbon-cands-%j-%t.err" \
    bash -c 'source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
        S=(42 43 44); s=${S[$SLURM_PROCID]}; P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 + SLURM_LOCALID * 4 ))
        start_vllm Qwen/Qwen3-4B-Instruct-2507 $P --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
            --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules "s$s$VARIANT=runs/$RUN/train/seed$s/adapter$VARIANT"
        NO_LENGTH_NOTE=1 "$VENV/bin/python" research/bestofn/candidates.py 0 1 "http://127.0.0.1:$P/v1" "s$s$VARIANT" \
            "runs/$OUT/s$s/data.jsonl" "runs/$OUT/s$s/${CDIR:-cands}" ${K:-3}'
