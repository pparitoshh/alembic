# Shared settings for every Leonardo job. Fill in the TODO lines once.
# Sourced by the submitting shell (slurm/submit.sh) and by every job script.

# --- Slurm: read by sbatch at submission (no hard-coded account in the scripts) ---
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-TODO_account}"        # project account: `saldo -b`
export SBATCH_PARTITION="${SBATCH_PARTITION:-boost_usr_prod}"   # Booster: 4x A100 64 GB per node
# export SBATCH_RESERVATION="TODO_hackathon_reservation"        # if the hackathon node is a reservation

# --- paths ($WORK is set by CINECA per project) ---
export REPO="${REPO:-$WORK/alembic}"                 # this repository
export VENV="${VENV:-$REPO/.venv}"                   # distillkit (uv sync --extra train --extra export)
export VLLM_VENV="${VLLM_VENV:-$WORK/venvs/vllm}"    # vLLM in its own venv: it pins its own torch
export LLAMA_CPP="${LLAMA_CPP:-$WORK/tools/llama.cpp}"
export LLAMA_CPP_TAG="b11392"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"          # model weights, downloaded on a login node

# --- models (downloaded by setup_login.sh) ---
export TEACHER_MODEL="Qwen/Qwen3-32B-AWQ"
export STUDENT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export JUDGE_MODEL="openai/gpt-oss-20b"
export CROSS_JUDGE_MODEL="google/gemma-4-26B-A4B-it"

# --- experiment ---
export CONFIG="${CONFIG:-configs/qwen3_4b_qdora.yaml}"
export RUN_DIR="${RUN_DIR:-runs/qwen3_4b_qdora}"     # shared generated/verified data; seeds in $RUN_DIR/seed_<n>
export SEEDS="${SEEDS:-42 1 2}"                       # >= 3 seeds per setting (GOAL.md §7)
