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
srun --kill-on-bad-exit=0 --output="$L/train-%j-%t.out" --error="$L/train-%j-%t.err" \
    bash -c 'source "$REPO/research/rag65/env.sh"; S=(42 43 44); exec "$VENV/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 "$REPO/research/rag65/train_seed.py" "$REPO/configs/${RUN:-rag65}_seed${S[$SLURM_NODEID]}.json"'
