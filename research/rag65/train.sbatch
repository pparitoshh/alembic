#!/bin/bash
#SBATCH --job-name=dk-rag65-train
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# Node i trains seed SEEDS[i] of configs/${RUN:-rag65}_seed*.json with 4-GPU DDP. Resubmit to continue from checkpoints.
: "${REPO:?}"
L="$REPO/../logs"
# NODES_PER_SEED=2 (sbatch --nodes=6): each seed trains with 8-GPU DDP over a node pair (c10d rendezvous on
# the pair's first node). Set the config's per-device batch so the global batch stays the same.
srun --kill-on-bad-exit=0 --output="$L/train-%j-%t.out" --error="$L/train-%j-%t.err" \
    bash -c 'source "$REPO/research/rag65/env.sh"; S=(42 43 44); NPS=${NODES_PER_SEED:-1}; PAIR=$(( SLURM_NODEID / NPS ))
        CFG="$REPO/configs/${RUN:-rag65}_seed${S[$PAIR]}.json"
        if [ "$NPS" = 1 ]; then
            exec "$VENV/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 "$REPO/research/rag65/train_seed.py" "$CFG"
        fi
        MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n "$(( PAIR * NPS + 1 ))p")
        exec "$VENV/bin/python" -m torch.distributed.run --nnodes="$NPS" --node-rank=$(( SLURM_NODEID % NPS )) --nproc-per-node=4 \
            --rdzv-backend=c10d --rdzv-endpoint="$MASTER:$(( 29500 + PAIR ))" --rdzv-id="$SLURM_JOB_ID-$PAIR" \
            "$REPO/research/rag65/train_seed.py" "$CFG"'
