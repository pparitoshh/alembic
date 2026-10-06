# Helpers for job scripts (sourced after env.sh, inside a Slurm job).

set -euo pipefail

# compute nodes: no internet assumed; everything comes from $HF_HOME and the venvs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# vLLM/FlashInfer runtime tools on Leonardo.
export PATH="$VLLM_VENV/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

cd "$REPO"
mkdir -p slurm/logs

distillkit() { "$VENV/bin/distillkit" "$@"; }

# A port no other job on this shared node uses: derived from the job (and array task) id.
job_port() {
    local offset="${1:-0}"
    echo $(( 20000 + (${SLURM_JOB_ID:-0} * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 3 + offset) % 20000 ))
}

# start_vllm <model> <port> [extra vllm args...]: serve in the background, wait until healthy.
# The server is stopped when the job script exits.
start_vllm() {
    local model="$1" port="$2"; shift 2
    local log="slurm/logs/vllm-${SLURM_JOB_ID:-local}-${port}.log"
    echo "[job] starting vLLM: $model on 127.0.0.1:$port (log: $log)"
    "$VLLM_VENV/bin/vllm" serve "$model" --host 127.0.0.1 --port "$port" "$@" >"$log" 2>&1 &
    VLLM_PIDS="${VLLM_PIDS:-} $!"
    trap 'stop_vllm' EXIT
    local waited=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
        if ! kill -0 "$!" 2>/dev/null; then
            echo "[job] vLLM exited during startup; last log lines:"; tail -n 40 "$log"; exit 1
        fi
        sleep 10; waited=$((waited + 10))
        if (( waited > 1800 )); then echo "[job] vLLM not ready after 30 min"; tail -n 40 "$log"; exit 1; fi
    done
    echo "[job] vLLM ready after ${waited}s"
}

stop_vllm() {
    for pid in ${VLLM_PIDS:-}; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    VLLM_PIDS=""
}

# vLLM flags per model family
teacher_args() { echo "--max-logprobs 20 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384 --gpu-memory-utilization 0.90"; }
judge_args() {
    case "$1" in
        google/gemma-4*) echo "--max-model-len 16384 --gpu-memory-utilization 0.90 --limit-mm-per-prompt {\"image\":0}" ;;
        *)               echo "--max-model-len 16384 --gpu-memory-utilization 0.85" ;;
    esac
}

seed_of_task() {  # SEEDS[$SLURM_ARRAY_TASK_ID]
    local seeds=($SEEDS)
    echo "${seeds[${SLURM_ARRAY_TASK_ID:-0}]}"
}
