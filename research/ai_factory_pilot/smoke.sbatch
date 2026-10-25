#!/bin/bash
#SBATCH --job-name=dk-ai-factory-smoke
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
source "${REPO:?Set isolated REPO}/research/ai_factory_pilot/environment.sh"
hostname
date -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
source "$REPO/research/ai_factory_pilot/gpu_preflight.sh"
"$VLLM_VENV/bin/python" - <<'PY'
import torch, vllm
assert torch.cuda.is_available()
x = torch.ones(256, 256, device='cuda', dtype=torch.bfloat16)
y = x @ x
assert torch.isfinite(y).all()
torch.cuda.synchronize()
print('VLLM_CUDA_KERNEL_OK', torch.__version__, vllm.__version__)
PY
# Launch and query the actual inference server, not just nvidia-smi.
PORT=$(job_port)
start_vllm Qwen/Qwen3-4B-Instruct-2507 "$PORT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85
"$VENV/bin/python" - "$PORT" <<'PY'
import json, sys, urllib.request
payload = {'model': 'Qwen/Qwen3-4B-Instruct-2507', 'messages': [{'role':'user','content':'Reply with OK.'}], 'max_tokens':16, 'temperature':0}
req = urllib.request.Request('http://127.0.0.1:'+sys.argv[1]+'/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=120) as response:
    data = json.load(response)
assert data['choices'][0]['message']['content'].strip()
print('VLLM_INFERENCE_OK', data['choices'][0]['message']['content'])
PY
stop_vllm
phase runtime_smoke_passed
