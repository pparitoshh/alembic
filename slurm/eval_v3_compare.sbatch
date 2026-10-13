#!/bin/bash
#SBATCH --job-name=dk-eval-v3-b6
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=slurm/logs/%x-%A_%a.out
# Submit afterok:<build job>. Existing B6 adapter bytes are only read.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$VENV/bin/python" -m distillkit.eval_v3 check-release
SEED=$(seed_of_task)
: "${B6_RUN:?Set B6_RUN to the existing three-seed B6 run directory}"
SEED_DIR="runs/eval_v3_b6/seed_$SEED"
ADAPTER="$B6_RUN/seed_$SEED/adapter"
test -s "$ADAPTER/adapter_model.safetensors"
mkdir -p "$SEED_DIR"
test -r "$ADAPTER/adapter_config.json" && test -r "$ADAPTER/adapter_model.safetensors"
# Do not silently run inference on CPUs if a checkout points to a CUDA 13
# environment on Leonardo's CUDA 12 driver. Use the repository's locked stack.
TORCH_DISABLE_NATIVE_JIT=1 "$VENV/bin/python" - "$SEED_DIR" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError(f"CUDA unavailable in {sys.executable} (torch {torch.__version__}, "
                       f"CUDA {torch.version.cuda}); provision the locked Linux environment before retrying")
probe = torch.empty(1, device="cuda")
torch.cuda.synchronize()
runtime = {"python": sys.version, "executable": sys.executable,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(),
           "packages": {name: importlib.metadata.version(name)
                        for name in ("transformers", "peft", "bitsandbytes", "accelerate")}}
(Path(sys.argv[1]) / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
print(json.dumps(runtime), flush=True)
PY
if [ ! -e "$SEED_DIR/adapter" ]; then ln -s "$ADAPTER" "$SEED_DIR/adapter"; fi
test "$(readlink -f "$SEED_DIR/adapter")" = "$(readlink -f "$ADAPTER")"
sha256sum "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" > "$SEED_DIR/adapter.before.sha256"
SET=(--set run_dir="$SEED_DIR" --set eval.file=data/eval/eval_v3.jsonl --set eval.tool_file=null)
distillkit answer -c configs/qwen3_4b_qdora.yaml "${SET[@]}"
PORT=$(job_port 1)
start_vllm "$JUDGE_MODEL" "$PORT" $(judge_args "$JUDGE_MODEL")
distillkit evaluate -c configs/qwen3_4b_qdora.yaml "${SET[@]}" \
    --set judge.base_url="http://127.0.0.1:$PORT/v1" --set judge.model="$JUDGE_MODEL" --set judge.temperature=0
stop_vllm
sha256sum --check "$SEED_DIR/adapter.before.sha256"
"$VENV/bin/python" - "$SEED_DIR" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = json.loads((p / 'eval_summary.json').read_text())
assert s['judge_protocol'] == 'paired-orders-v2-invalid-incomplete'
assert s['judge'] == 'openai/gpt-oss-20b'
c = s['judge_comparisons']['student_vs_base']
assert c['status'] == 'complete' and c['questions'] == c['complete_pairs'] == 200
assert c['invalid_verdicts'] == 0
assert s['student_vs_base_win_rate'] is not None
PY
