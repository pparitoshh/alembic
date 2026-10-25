#!/bin/bash
#SBATCH --job-name=dk-agent-collect
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
# Agent trajectories on real Slurm (research/agent_train/collect.py): every node serves the teacher on its 4 GPUs
# (one vLLM each; the first load reads Lustre, the other three then load from the page cache in parallel) and runs
# 4 collectors (8 agents each). OUT, TEACHER (path or HF id), N (tasks per collector).
source "${REPO:?}/research/rag65/env.sh"
source "$REPO/slurm/common.sh"  # VLLM_USE_FLASHINFER_SAMPLER=0 (no nvcc on the nodes), offline HF, PATH
export OUT=${OUT:?} TEACHER=${TEACHER:-Qwen/Qwen3-32B-AWQ} N=${N:-40}
srun --ntasks-per-node=1 bash -c '
    cd "$REPO"; export PYTHONPATH=src; P=$(( 21000 + (SLURM_JOB_ID % 400) * 20 )); NODE=$SLURM_NODEID
    serve() { CUDA_VISIBLE_DEVICES=$1 "$VLLM_VENV/bin/vllm" serve "$TEACHER" --host 127.0.0.1 --port $((P + $1)) \
        --served-model-name teacher --dtype auto --max-model-len 32768 --gpu-memory-utilization 0.88 \
        --enable-auto-tool-choice --tool-call-parser hermes > "$OUT/vllm-$SLURM_JOB_ID-$NODE-$1.log" 2>&1 & }
    ready() { until curl -sf http://127.0.0.1:$((P + $1))/health > /dev/null; do sleep 5; done; }
    serve 0; ready 0; for g in 1 2 3; do serve $g; done; for g in 1 2 3; do ready $g; done
    echo "node $NODE: 4 teachers ready at $(date +%T)"
    for g in 0 1 2 3; do
        "$VENV/bin/python" research/agent_train/collect.py "$OUT" EUHPC_D30_031 lrd_all_serial runs/rag65/indexes \
            http://127.0.0.1:$((P + g))/v1 teacher "$N" "$((SLURM_JOB_ID % 100000 * 16 + NODE * 4 + g))" 8 \
            > "$OUT/collect-$SLURM_JOB_ID-$NODE-$g.log" 2>&1 &
    done
    wait $(jobs -p | tail -4)
    pkill -P $$ -f "vllm serve"
'
