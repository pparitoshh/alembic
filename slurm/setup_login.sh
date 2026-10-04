#!/bin/bash
# One-time setup on a Leonardo LOGIN node (has internet; compute nodes are assumed not to).
#   git clone <repo> $WORK/alembic && cd $WORK/alembic && bash slurm/setup_login.sh
# Idempotent: rerun after changing models or versions.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"

# uv (user install, no modules needed)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# distillkit venv: Python 3.12 managed by uv
uv sync --extra train --extra export

# vLLM in its own venv (it pins torch/transformers versions of its own)
[ -x "$VLLM_VENV/bin/vllm" ] || { uv venv -p 3.12 "$VLLM_VENV" && uv pip install -p "$VLLM_VENV/bin/python" vllm; }
"$VLLM_VENV/bin/vllm" --version

# model weights into $HF_HOME (teacher ~19 GB, student ~8 GB, judges ~14 GB + ~52 GB)
for m in "$TEACHER_MODEL" "$STUDENT_MODEL" "$JUDGE_MODEL" "$CROSS_JUDGE_MODEL"; do
    echo "[setup] downloading $m"
    "$VENV/bin/hf" download "$m"
done

# llama.cpp for export: source tag (converter) + CPU release binaries (no compiler needed)
if [ ! -x "$LLAMA_CPP/bin/llama-quantize" ]; then
    git clone --depth 1 --branch "$LLAMA_CPP_TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
    mkdir -p "$LLAMA_CPP/bin"
    curl -L "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_CPP_TAG/llama-$LLAMA_CPP_TAG-bin-ubuntu-x64.tar.gz" \
        | tar -xz -C "$LLAMA_CPP/bin" --strip-components=1
fi

# hermetic test suite as a final check of the install
"$VENV/bin/python" -m pytest -q
echo "[setup] done. Quota check: cindata (or du -sh $HF_HOME)"
