#!/bin/bash
# CPU-only (laptop-like) end-to-end check of the integrated assistant:
# llama.cpp build -> merge a scaled adapter -> GGUF bf16 -> Q4_K_M -> llama-server -> hpc-assistant with built-in index.
# usage: sbatch -p lrd_all_serial -c 4 --mem=30G -t 04:00:00 build_and_smoke.sh ADAPTER_DIR OUT_DIR
set -euo pipefail
ADAPTER=$(realpath "$1"); OUT=$(realpath -m "$2")
R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017
C=$R/code
V=/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z/runtime/train-cu129/bin/python
LLAMA=$R/llama.cpp
LLAMA_COMMIT=5aeac37a35b44fde4bdffaec9159f2fcb253b622
export HF_HOME=/leonardo_work/EUHPC_D30_031/hf_cache HF_HUB_OFFLINE=1
mkdir -p "$OUT"
step() { echo "$(date -u +%FT%TZ) $*"; }

if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
    step "building llama.cpp $LLAMA_COMMIT"
    [ -d "$LLAMA/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA"
    git -C "$LLAMA" fetch -q --depth 1 origin "$LLAMA_COMMIT" && git -C "$LLAMA" checkout -q "$LLAMA_COMMIT"
    module load cmake/3.27.9 2>/dev/null || true
    # System gcc 8.5 lacks std::filesystem in libstdc++; use spack's gcc 12 with a static C++ runtime so the
    # binaries need no extra library path.
    GCC=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/gcc-12.2.0-lkcazt4letxjj4s7nlhzryoyivevsatz
    rm -rf "$LLAMA/build"
    CC=$GCC/bin/gcc CXX=$GCC/bin/g++ cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" > "$OUT/cmake.log" 2>&1
    cmake --build "$LLAMA/build" -j 4 --target llama-quantize llama-server >> "$OUT/cmake.log" 2>&1
fi

if [ ! -f "$OUT/merged/config.json" ]; then
    step "merging $ADAPTER"
    "$V" - "$ADAPTER" "$OUT/merged" <<'PY'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
adapter, out = sys.argv[1], sys.argv[2]
base = 'Qwen/Qwen3-4B-Instruct-2507'
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
PY
fi
[ -f "$OUT/model-bf16.gguf" ] || { step "converting"; PYTHONPATH="$LLAMA/gguf-py" "$V" "$LLAMA/convert_hf_to_gguf.py" "$OUT/merged" --outtype bf16 --outfile "$OUT/model-bf16.gguf" > "$OUT/convert.log" 2>&1; }
[ -f "$OUT/model-Q4_K_M.gguf" ] || { step "quantizing"; "$LLAMA/build/bin/llama-quantize" "$OUT/model-bf16.gguf" "$OUT/model-Q4_K_M.gguf" Q4_K_M > "$OUT/quantize.log" 2>&1; }
ls -la "$OUT"/*.gguf

step "smoke test: CPU llama-server (4 threads) + hpc-assistant with built-in index"
PORT=$((20000 + RANDOM % 20000))
"$LLAMA/build/bin/llama-server" -m "$OUT/model-Q4_K_M.gguf" -t 4 -c 8192 --jinja --port $PORT --host 127.0.0.1 > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 120); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 5; done
cd "$C"
for mode in docs no-docs; do
    for q in "How do I request 2 GPUs for 4 hours on LUMI with Slurm?" "Which partition do I use for GPU jobs on Leonardo, and how many GPUs does a node have?"; do
        flag=; [ $mode = no-docs ] && flag=--no-docs
        start=$(date +%s.%N)
        PYTHONPATH=src "$V" -m distillkit.assistant $flag --index runs/rag65/indexes --openai "http://127.0.0.1:$PORT/v1" "$q" > "$OUT/answer.tmp" 2>&1
        secs=$(echo "$(date +%s.%N) - $start" | bc)
        printf '\n=== [%s] %.1fs: %s\n' "$mode" "$secs" "$q" | tee -a "$OUT/smoke.txt"
        cat "$OUT/answer.tmp" >> "$OUT/smoke.txt"
        head -c 600 "$OUT/answer.tmp"; echo
    done
done
rm -f "$OUT/answer.tmp" "$OUT/model-bf16.gguf"
rm -rf "$OUT/merged"
step "SMOKE_OK"
