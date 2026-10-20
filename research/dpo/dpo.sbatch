#!/bin/bash
#SBATCH --job-name=dk-rag65dpo-train
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# DPO on 2 nodes (8-GPU DDP, c10d rendezvous on the first node). Resubmit to continue from checkpoints.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/dpo-%j-%t.out" --error="$L/dpo-%j-%t.err" \
    bash -c 'source "$REPO/research/rag65/env.sh"; MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
        exec "$VENV/bin/python" -m torch.distributed.run --nnodes=2 --node-rank=$SLURM_NODEID --nproc-per-node=4 \
            --rdzv-backend=c10d --rdzv-endpoint="$MASTER:29600" --rdzv-id="$SLURM_JOB_ID" \
            "$REPO/research/dpo/train_dpo.py" "$REPO/runs/dpo/merged_r5" "$REPO/runs/dpo/train.jsonl" "$REPO/runs/dpo/out"'
