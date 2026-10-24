#!/bin/bash
#SBATCH --job-name=dk-scale
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# GOAL.md G11 scaling test: the same LoRA fine-tuning of Qwen3-4B (research/kd/train_kd.py, plain loss) as 4-GPU DDP on
# one node, or DDP over all nodes of the job (c10d rendezvous on the first node), for a fixed wall time (TRAIN_SECONDS),
# then stopped. The per-step time in the log gives samples/s: each optimizer step is 16 x nodes samples
# (per-device batch 1, gradient accumulation 4 per GPU). sbatch --nodes=N --export=ALL,CFG=...,TRAIN_SECONDS=720
: "${REPO:?}"
L="$REPO/../logs"
srun --kill-on-bad-exit=0 --output="$L/scale-%j-%t.out" --error="$L/scale-%j-%t.err" bash -c '
    source "$REPO/research/rag65/env.sh"; cd "$REPO"; export PYTHONPATH=src
    MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
    timeout ${TRAIN_SECONDS:-720} "$VENV/bin/python" -m torch.distributed.run --nnodes=$SLURM_NNODES --node-rank=$SLURM_NODEID \
        --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29511 --rdzv-id=$SLURM_JOB_ID \
        research/kd/train_kd.py "${CFG:?}" /nonexistent 0
    echo "[scale] stopped after ${TRAIN_SECONDS:-720} s on $SLURM_NNODES node(s)"'
