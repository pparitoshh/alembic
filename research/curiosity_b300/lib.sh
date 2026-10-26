# Shared helpers for Curiosity tasks (sourced). One 8-GPU node; tasks start and stop their own vLLM servers.
CP=$HOME/cp; REPO=$CP/alembic; M=$CP/models
source $CP/venv/bin/activate
# CUDA 13 toolkit from the venv (matches torch cu130) for FlashInfer / DeepGEMM JIT kernels
export CUDA_HOME=$CP/venv/lib/python3.12/site-packages/nvidia/cu13 CUDA_PATH=$CP/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-} LIBRARY_PATH=$CUDA_HOME/lib:${LIBRARY_PATH:-}
export HF_HOME=$CP/hf HF_HUB_OFFLINE=1 PYTHONPATH=$REPO/src:$REPO/research/rag65 VLLM_NO_USAGE_STATS=1
log() { echo "$(date -u +%FT%TZ) [${TASK:-cp}] $*" | tee -a $CP/logs/tasks.log; }
PIDS=()
# serve GPUS PORT MODEL SERVED_NAME [vllm args...]: start a vLLM server on the given GPUs, wait until it answers
serve() {
    local gpus=$1 port=$2 model=$3 name=$4; shift 4
    local tp=$(awk -F, '{print NF}' <<< "$gpus")
    CUDA_VISIBLE_DEVICES=$gpus nohup vllm serve "$model" --port $port --served-model-name "$name" \
        --tensor-parallel-size $tp --dtype auto "$@" > $CP/logs/vllm_${TASK:-x}_${name//\//_}_$port.log 2>&1 &
    PIDS+=($!)
    for i in $(seq 1 180); do
        curl -sf http://127.0.0.1:$port/v1/models > /dev/null && { log "server $name on GPUs $gpus port $port ready"; return 0; }
        kill -0 ${PIDS[-1]} 2>/dev/null || { log "server $name died (see logs/vllm_${TASK:-x}_${name//\//_}_$port.log)"; return 1; }
        sleep 10
    done
    log "server $name not ready after 30 min"; return 1
}
# Tasks run one at a time on the node, so at the end of a task every vLLM server and engine of this user goes
# (servers started in the background with `serve ... &` are not in this shell's PIDS; Oct 26 fix).
stop_all() {
    for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done
    pkill -u $USER -f "^$CP/venv/bin/python3 $CP/venv/bin/vllm serve" 2>/dev/null; sleep 5
    pkill -9 -u $USER -f "^$CP/venv/bin/python3 $CP/venv/bin/vllm serve" 2>/dev/null
    pkill -9 -u $USER -f "^VLLM::" 2>/dev/null; PIDS=()
}
stop_all  # tasks run one at a time: nothing from an earlier task may hold a GPU
trap stop_all EXIT
# wait_models: block until the model downloads (dl.sh, gpt-oss-20b) have finished
wait_models() {
    until grep -q ALL_DONE $CP/dl.log 2>/dev/null && grep -q "OK openai/gpt-oss-20b" $CP/dl.log; do sleep 60; done
    grep -q FAIL $CP/dl.log && { log "STOP: a model download failed: $(grep FAIL $CP/dl.log)"; return 1; }
    return 0
}
