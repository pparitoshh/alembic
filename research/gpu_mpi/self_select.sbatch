#!/bin/bash
#SBATCH --job-name=dk-selfselect
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
# Self-selection test: round 27 picks between its with-excerpts and closed-book answers (research: GPU/MPI push).
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
P=$(( 23000 + (SLURM_JOB_ID % 400) * 10 ))
start_vllm /leonardo_work/EUHPC_D30_031/shared_split/models/r27_merged $P --served-model-name r27 --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.88
"$VENV/bin/python" "$REPO/../ctl/self_select.py" "http://127.0.0.1:$P/v1" r27 "$REPO/runs/gate27/self_select.json"
