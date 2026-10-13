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

# Index of this seed task: the srun task id in whole-node jobs (fan_out_seeds), or the array task id.
task_index() { echo "${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}"; }

# A port no other job or task on this node uses: derived from the job id and the task index.
job_port() {
    local offset="${1:-0}"
    echo $(( 20000 + (${SLURM_JOB_ID:-0} * 7 + $(task_index) * 3 + offset) % 20000 ))
}

# fan_out_seeds <script>: whole-node jobs. In the batch step, start one srun task per seed in $SEEDS
# (1 GPU, $SLURM_CPUS_PER_TASK cores each) that runs <script> again, then exit with srun's status.
# Inside a task (or in a legacy array task) it returns, and the script continues for one seed.
# Per-seed logs: slurm/logs/<job name>-<job id>_seed<task>.out
fan_out_seeds() {
    if [ -n "${DK_SEED_TASK:-}" ] || [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then return 0; fi
    local n; n=$(wc -w <<< "$SEEDS")
    DK_SEED_TASK=1 srun --ntasks="$n" --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --exact \
        --output="slurm/logs/${SLURM_JOB_NAME:-dk}-${SLURM_JOB_ID:-0}_seed%t.out" bash "$REPO/$1"
    exit $?
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

seed_of_task() {  # SEEDS[task index]
    local seeds=($SEEDS)
    echo "${seeds[$(task_index)]}"
}
