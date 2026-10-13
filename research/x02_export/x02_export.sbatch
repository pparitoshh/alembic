#!/bin/bash
#SBATCH --job-name=dk-x02-export
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
# x0.2 (B7 seed 42 at 20%) -> merged/ -> GGUF Q4_K_M (no imatrix: CPU-only llama.cpp) + Modelfile.
# Usage (from the repo root on Leonardo): LLAMA_CPP=$WORK/tools/llama.cpp REPO=$PWD sbatch research/x02_export/x02_export.sbatch
set -e
source "$REPO/slurm/env.sh"; source "$REPO/slurm/common.sh"
RD=runs/x02_s42/seed_42; mkdir -p $RD/export
ln -sfn ../../b7_hpc10k/verified.jsonl $RD/verified.jsonl
[ -f $RD/export/merged/config.json ] || "$VENV/bin/python" research/x02_export/scaled_merge.py runs/b7_hpc10k/seed_42/adapter $RD/export/merged 0.2
distillkit export -c configs/qwen3_4b_qdora.yaml --set run_dir=$RD --set export.llama_cpp="$LLAMA_CPP" \
    --set export.ollama_name=null --set export.imatrix=false --set 'export.quants=[Q4_K_M]'
ls -la $RD/export
