# Shared settings for the fix 4 (multi-step tool use) pipeline on Curiosity (sourced).
# Two roots: CP_RO (team folder, read only: venv, models, 1.3 student and data, eval data) and WORK (all writes).
# It does not source the team lib.sh, because that lib.sh writes under the team folder and stops every vLLM of the user.
# env: NAME (default v1), SEED (default 42), TEACHER (q235 | gptoss120, default q235), SMOKE (1 = small test run)
CP_RO=${CP_RO:-/home/eai-jjppv/cp}
WORK=${WORK:-$HOME/mt}
REPO=${REPO:-$WORK/alembic}
CP=$CP_RO; M=$CP_RO/models                 # eval.sh reads $CP/data and $M through these names
export NGPU=${NGPU:-4}                     # this pipeline uses the 4 GPUs that the agent job does not use
source $CP_RO/venv/bin/activate
# CUDA 13 toolkit from the venv (matches torch cu130) for FlashInfer / DeepGEMM JIT kernels (as curiosity_b300/lib.sh)
export CUDA_HOME=$CP_RO/venv/lib/python3.12/site-packages/nvidia/cu13 CUDA_PATH=$CP_RO/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-} LIBRARY_PATH=$CUDA_HOME/lib:${LIBRARY_PATH:-}
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1 PYTHONPATH=$REPO/src:$REPO/research/rag65 VLLM_NO_USAGE_STATS=1
mkdir -p $WORK/logs $WORK/exp $WORK/runs $HF_HOME
# answer.py opens the retrieval indexes at the relative path runs/rag65/indexes (cwd $REPO); the rsync skips runs/
mkdir -p $REPO/runs/rag65; [ -e $REPO/runs/rag65/indexes ] || ln -s $CP_RO/data/rag65/indexes $REPO/runs/rag65/indexes
LOGS=$WORK/logs
log() { echo "$(date -u +%FT%TZ) [${TASK:-mt}] $*" | tee -a $LOGS/tasks.log; }
wait_models() { return 0; }                # eval.sh calls it; the team models are already complete

NAME=${NAME:-v1}; SEED=${SEED:-42}; TEACHER=${TEACHER:-q235}; SMOKE=${SMOKE:-0}
[ "$SMOKE" = 1 ] && [[ $NAME != *_smoke ]] && NAME=${NAME}_smoke
TASK=${TASK:-mt_$NAME}
export ACCUM=${ACCUM:-$(( 64 / NGPU ))}    # 64 sequences per optimizer step, as exp_kd.sh and kdall.sh
MT=$REPO/research/train_multistep_tools    # Python tools of the pipeline
AT=$REPO/research/agent_train
CB=$REPO/research/curiosity_b300           # train_kd.py, eval.sh, answer.py from the repo checkout
RUN=$WORK/exp/mt_$NAME                     # shared data of all seeds: tasks, trajectories, examples, parts
OUT=$WORK/exp/mt_${NAME}_s$SEED            # one trained model per seed
STUDENT=${STUDENT:-$CP_RO/exp/kdall/model} # ClusterPilot 1.3
KDALL=$CP_RO/exp/kdall                     # 1.3 teacher data (prompts.jsonl, teacher/*.jsonl)
LIMIT=${LIMIT:-}; [ "$SMOKE" = 1 ] && LIMIT=${LIMIT:-50}
case $TEACHER in
    q235)      TEACHER_DIR=$M/q235fp8;   TEACHER_NAME=qwen3-235b;   TOOL_PARSER=hermes ;;
    gptoss120) TEACHER_DIR=$M/gptoss120; TEACHER_NAME=gpt-oss-120b; TOOL_PARSER=openai ;;
    *) echo "unknown TEACHER=$TEACHER (q235 | gptoss120)" >&2; exit 2 ;;
esac
PORT=${PORT:-8200}; ENDPOINT=http://127.0.0.1:$PORT/v1
mkdir -p $RUN

# Servers: each started PID goes to a file, so servers started in background subshells (eval.sh) are stopped too.
# stop_all stops only these PIDs (and their children), never other vLLM servers of the user.
export MT_PIDFILE=${MT_PIDFILE:-$LOGS/pids.$$}
# serve GPUS PORT MODEL SERVED_NAME [vllm args...]: start a vLLM server, wait until /health answers
serve() {
    local gpus=$1 port=$2 model=$3 name=$4; shift 4
    local tp=$(awk -F, '{print NF}' <<< "$gpus") lf=$LOGS/vllm_${TASK:-x}_${name//\//_}_$port.log
    CUDA_VISIBLE_DEVICES=$gpus setsid nohup vllm serve "$model" --port $port --served-model-name "$name" \
        --tensor-parallel-size $tp --dtype auto "$@" > $lf 2>&1 &
    local pid=$!; echo $pid >> $MT_PIDFILE
    for i in $(seq 1 180); do
        curl -sf http://127.0.0.1:$port/health > /dev/null && { log "server $name on GPUs $gpus port $port ready"; return 0; }
        kill -0 $pid 2>/dev/null || { log "server $name died (see $lf)"; return 1; }
        sleep 10
    done
    log "server $name not ready after 30 min"; return 1
}
stop_all() {
    [ -s $MT_PIDFILE ] || return 0
    for p in $(cat $MT_PIDFILE); do kill -- -$p 2>/dev/null || kill $p 2>/dev/null; done; sleep 5
    for p in $(cat $MT_PIDFILE); do kill -9 -- -$p 2>/dev/null || kill -9 $p 2>/dev/null; done
    : > $MT_PIDFILE
}
_mt_exit() { [ "$MT_PIDFILE" = "$LOGS/pids.$$" ] && { stop_all; rm -f $MT_PIDFILE; }; }  # only the top shell cleans up
trap _mt_exit EXIT
# eval.sh sources this file again: it then keeps the PID file of its caller (MT_PIDFILE is exported)
export EVAL_LIB=${BASH_SOURCE[0]} EVAL_RUNS=$WORK/runs EVAL_LOGS=$LOGS EVAL_BIN=$CB
