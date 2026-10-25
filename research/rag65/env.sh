set -euo pipefail
: "${REPO:?}"
export VENV=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129
export VLLM_VENV=/leonardo_work/EUHPC_D30_031/venvs/vllm
export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PYTHONPATH="$REPO/src:$REPO/research/rag65"
export TORCH_DISABLE_NATIVE_JIT=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PATH="$VLLM_VENV/bin:$PATH"
cd "$REPO"
