set -euo pipefail
: "${REPO:?Set isolated REPO at submission}"
export VENV=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129
export VLLM_VENV=/leonardo_work/EUHPC_D30_031/venvs/vllm
export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache
export PYTHONPATH="$REPO/src"
export TORCH_DISABLE_NATIVE_JIT=1
export CONFIG="$REPO/configs/ai_factory_pilot.json"
export RUN_DIR="$REPO/runs/ai_factory_pilot_20261013"
export OMP_NUM_THREADS=8
source "$REPO/slurm/common.sh"
# Use this checkout's code, without changing any shared environment or another user's checkout.
distillkit() { "$VENV/bin/python" -m distillkit.cli "$@"; }
phase() { printf '%s %s job=%s task=%s\n' "$(date -u +%FT%TZ)" "$1" "${SLURM_JOB_ID:-local}" "${SLURM_ARRAY_TASK_ID:-none}" | tee -a "$RUN_DIR/pipeline-${SLURM_JOB_ID:-local}.log"; }
