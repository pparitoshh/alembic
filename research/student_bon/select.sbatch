#!/bin/bash
#SBATCH --job-name=dk-rag65-sbon
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# The reviewer picks the best of the student's greedy answer + the samples. One node per seed (42, 43, 44), so the
# three seeds are reviewed at the same time. SELECTOR=gemma (Gemma 4 26B, default) or tnext (Qwen3-Next-80B-A3B
# AWQ); SETS="cands:select ..." pairs of candidate and selection directories under runs/$OUT/s<seed>.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/sbon-select-%j-%t.out" --error="$L/sbon-select-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; source "$REPO/slurm/common.sh"; cd "$REPO"; export PYTHONPATH=src:research/rag65
    S=(42 43 44); s=${S[$SLURM_PROCID]}; P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
    SO="{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}"
    if [ "${SELECTOR:-gemma}" = tnext ]; then
        start_vllm cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit $P --tensor-parallel-size 4 --max-model-len 32768 \
            --gpu-memory-utilization 0.90 --structured-outputs-config "$SO" --served-model-name tnext
    else
        start_vllm google/gemma-4-26B-A4B-it $P --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.92 \
            --limit-mm-per-prompt "{\"image\":0}" --structured-outputs-config "$SO" --served-model-name gemma
    fi
    for pair in ${SETS:-cands:select}; do
        "$VENV/bin/python" research/bestofn/select.py 0 1 "http://127.0.0.1:$P/v1" "${SELECTOR:-gemma}" runs/$OUT/s$s/data.jsonl \
            runs/$OUT/s$s/${pair%%:*} runs/$OUT/s$s/${pair##*:}
    done'
