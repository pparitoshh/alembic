#!/bin/bash
#SBATCH --job-name=dk-gguf-imatrix
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Importance matrix from the round-5 training conversations (bf16 GGUF on one A100), then Q4_K_M and Q3_K_M
# re-quantized with it. Outputs runs/gguf/imatrix.gguf and model-Q4_K_M-imat.gguf / model-Q3_K_M-imat.gguf.
set -euo pipefail
cd "${REPO:?}"
module load cuda/12.6 > /dev/null 2>&1
B=/leonardo_work/EUHPC_D30_031/runs/a08trc09/published-hf-comparison-20261015/llama_cuda/bin
export LD_LIBRARY_PATH=$B:${LD_LIBRARY_PATH:-}
O=${OUT:-runs/gguf}  # GGUF directory (default: round 5 export)
[ -s $O/calibration.txt ] || python3 research/gguf/calibration.py runs/rag65v5/train/seed43/verified.jsonl configs/rag65v5_seed43.json $O/calibration.txt 400
[ -s $O/imatrix.gguf ] || $B/llama-imatrix -m $O/model-bf16.gguf -f $O/calibration.txt -o $O/imatrix.gguf -ngl 99 -c 2048 -b 2048 --chunks 200
for q in Q4_K_M Q3_K_M; do
    [ -s $O/model-$q-imat.gguf ] || $B/llama-quantize --imatrix $O/imatrix.gguf $O/model-bf16.gguf $O/model-$q-imat.gguf $q 8
done
ls -la $O
