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
# Best-of-n ceiling probe on the eval sets. PHASE=cands: TEACHER (tnext TP4 or t32) writes 3 answers per question.
# PHASE=select: Gemma (TP2) picks the best. Output under runs/teacher_probe3/bon. Resubmit to resume.
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH=src:research/rag65
O=runs/teacher_probe3/bon
P=$(( 20000 + (SLURM_JOB_ID % 900) * 20 ))
if [ "${PHASE:?}" = cands ]; then
    if [ "${TEACHER:?}" = tnext ]; then
        start_vllm cyankiwi/Qwen3-Next-80B-A3B-Instruct-AWQ-4bit $P --tensor-parallel-size 4 --max-model-len 16384 \
            --gpu-memory-utilization 0.90 --served-model-name tnext
    else
        start_vllm Qwen/Qwen3-32B-AWQ $P --tensor-parallel-size 4 --max-model-len 16384 --gpu-memory-utilization 0.90 --served-model-name t32
    fi
    "$VENV/bin/python" research/bestofn/candidates.py 0 1 "http://127.0.0.1:$P/v1" "$TEACHER" $O/data.jsonl $O/cands 3
else
    start_vllm google/gemma-4-26B-A4B-it $P --tensor-parallel-size 4 --max-model-len 32768 --gpu-memory-utilization 0.92 \
        --limit-mm-per-prompt '{"image":0}' --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
        --served-model-name gemma
    "$VENV/bin/python" research/bestofn/select.py 0 1 "http://127.0.0.1:$P/v1" gemma $O/data.jsonl $O/cands $O/select
fi
