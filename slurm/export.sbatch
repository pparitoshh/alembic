#!/bin/bash
#SBATCH --job-name=dk-export
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm/logs/%x-%j.out
# Merge -> GGUF -> imatrix -> quants for one seed (EXPORT_SEED, default: first of $SEEDS).
# CPU work; the GPU is requested only because Booster jobs are GPU jobs (TODO: confirm a CPU-only option).
# No Ollama on the cluster: copy $RUN_DIR/seed_<n>/export/ to the laptop and run `ollama create` there.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"

SEED="${EXPORT_SEED:-$(echo $SEEDS | cut -d' ' -f1)}"
distillkit export -c "$CONFIG" --set run_dir="$RUN_DIR/seed_$SEED" --set student.model="$STUDENT_MODEL" \
    --set export.llama_cpp="$LLAMA_CPP" --set export.ollama_name=null
