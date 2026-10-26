#!/bin/bash
# Model downloads for Curiosity (round 27 = ClusterPilot-4B, base student, teacher/judge models).
source ~/cp/venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=~/cp/hf
M=~/cp/models; mkdir -p $M
dl() { hf download "$1" --local-dir "$M/$2" ${3:+--include "$3"} > ~/cp/dl_$2.log 2>&1 && echo "$(date -u +%T) OK $1" >> ~/cp/dl.log || echo "$(date -u +%T) FAIL $1 (see dl_$2.log)" >> ~/cp/dl.log; }
dl VaibhavKKM/hpc-assistant-qwen3-4b r27 "*.json" ; dl VaibhavKKM/hpc-assistant-qwen3-4b r27 "*.safetensors"; dl VaibhavKKM/hpc-assistant-qwen3-4b r27 "*.jinja"
dl Qwen/Qwen3-4B-Instruct-2507 qwen3-4b
dl Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 q235fp8 &
dl openai/gpt-oss-120b gptoss120 &
dl google/gemma-4-26B-A4B-it gemma4 &
wait; echo "$(date -u +%T) ALL_DONE" >> ~/cp/dl.log
