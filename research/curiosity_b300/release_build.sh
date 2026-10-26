#!/bin/bash
# Build the release folder for the model named in ~/cp/release/READY: bf16 weights, GGUF (Q8_0, Q6_K, Q4_K_M),
# Modelfile and the evaluation result. No token is used here; the laptop's release watcher uploads the folder.
set -euo pipefail
source $HOME/cp/venv/bin/activate
R=$HOME/cp/release; NAME=$(python3 -c "import json;print(json.load(open('$R/READY'))['name'])")
MODEL=$(python3 -c "import json;print(json.load(open('$R/READY'))['model'])")
O=$R/$NAME; mkdir -p $O/gguf
cp $MODEL/*.safetensors $MODEL/*.json $O/ 2>/dev/null; cp $MODEL/chat_template.jinja $O/ 2>/dev/null || true
rm -f $O/KD_TRAIN.json; cp $MODEL/KD_TRAIN.json $O/eval_train_recipe.json 2>/dev/null || true
cp $R/READY $O/eval_curiosity_result.json
L=$HOME/cp/llama.cpp
[ -s $O/gguf/hpc-assistant-qwen3-4b-bf16.gguf ] || python $L/convert_hf_to_gguf.py $MODEL --outtype bf16 --outfile $O/gguf/hpc-assistant-qwen3-4b-bf16.gguf
for q in Q8_0 Q6_K Q4_K_M; do
    [ -s $O/gguf/hpc-assistant-qwen3-4b-$q.gguf ] || $L/build/bin/llama-quantize $O/gguf/hpc-assistant-qwen3-4b-bf16.gguf $O/gguf/hpc-assistant-qwen3-4b-$q.gguf $q 32 > /dev/null
done
rm -f $O/gguf/hpc-assistant-qwen3-4b-bf16.gguf
[ -s $HOME/cp/release/Modelfile ] && sed "1s/.*/# Ollama package for the HPC assistant (ClusterPilot-4B, $NAME: round 27 further distilled from Qwen3-235B on B300s; Q4_K_M)./" $HOME/cp/release/Modelfile > $O/gguf/Modelfile
ls -la $O $O/gguf
echo "BUILD READY $NAME"
