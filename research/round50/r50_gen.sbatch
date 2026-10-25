#!/bin/bash
#SBATCH --job-name=dk-r50-gen
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
# Round 50 step 2: round 27 answers every question with and without excerpts (4 vLLM servers, one per GPU).
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"
P=$(( 24000 + (SLURM_JOB_ID % 400) * 10 )); EP=""
for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g start_vllm /leonardo_work/EUHPC_D30_031/shared_split/models/r27_merged $((P + g)) \
        --served-model-name r27 --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.88
    EP="$EP,http://127.0.0.1:$((P + g))/v1"
done
"$VENV/bin/python" "$REPO/../ctl/r50_gen_two.py" "${RD:?}" "${EP#,}" r27
