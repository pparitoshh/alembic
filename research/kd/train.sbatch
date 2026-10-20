#!/bin/bash
#SBATCH --job-name=dk-rag65kd-train
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# sbatch --nodes=1..3: node i trains seed SEEDS[i] of configs/${CFG:-rag65kd}_seed*.json with research/kd/train_kd.py
# (4-GPU DDP on one node). Resubmit to continue from checkpoints.
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --output="$L/train-%j-%t.out" --error="$L/train-%j-%t.err" \
    bash -c 'source "$REPO/research/rag65/env.sh"; cd "$REPO"; export PYTHONPATH=src; S=(42 43 44)
        exec "$VENV/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 research/kd/train_kd.py \
            "configs/${CFG:-rag65kd}_seed${S[$SLURM_NODEID]}.json" "${KD_DIR:?}"'
