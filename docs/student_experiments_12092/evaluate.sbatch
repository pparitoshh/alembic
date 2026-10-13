#!/bin/bash
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/student-experiments-v1
#SBATCH --output=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/student-experiments-v1/logs/%x-%j.out
#SBATCH --export=NONE
set -euo pipefail
BASE=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z
EXP="$BASE/student-experiments-v1"
VARIANT="$1"
export REPO="$EXP/code"
export VENV="$BASE/runtime/train-cu129"
export VLLM_VENV=/leonardo_work/EUHPC_D30_031/venvs/vllm
export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache
export PYTHONPATH="$REPO/src"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$EXP/runtime-$SLURM_JOB_ID"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TMPDIR="$XDG_CACHE_HOME/tmp"
export VLLM_RPC_BASE_PATH="/tmp/dkeval$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$VLLM_RPC_BASE_PATH"
source "$REPO/slurm/common.sh"
"$VENV/bin/python" -B "$EXP/evaluate_run.py" answers "$VARIANT"
PORT=$(job_port)
test "$(cat "$HF_HOME/hub/models--openai--gpt-oss-20b/refs/main")" = 6cee5e81ee83917806bbde320786a8fb61efebee
start_vllm openai/gpt-oss-20b "$PORT" $(judge_args openai/gpt-oss-20b) --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
"$VENV/bin/python" -B "$EXP/evaluate_run.py" judge "$VARIANT" --judge oss --port "$PORT"
stop_vllm
test "$(cat "$HF_HOME/hub/models--google--gemma-4-26B-A4B-it/refs/main")" = 4d7ae4984b7db7de8f8457170b3f1a419ee76d52
start_vllm google/gemma-4-26B-A4B-it "$PORT" --max-model-len 8192 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0}'
"$VENV/bin/python" -B "$EXP/evaluate_run.py" judge "$VARIANT" --judge gemma --port "$PORT"
stop_vllm
