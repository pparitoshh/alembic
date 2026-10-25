#!/bin/bash
#SBATCH --job-name=dk-ccby
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# 32 shards, one GPU each. Submit with PHASE=generate, then PHASE=review (afterok).
L="${REPO:?}/../logs"
srun --kill-on-bad-exit=0 --gpu-bind=single:1 --gpus-per-task=1 \
    --output="$L/${PHASE:?}-%j-%t.out" --error="$L/${PHASE}-%j-%t.err" \
    bash "$REPO/research/ai_factory_pilot/ccby_worker.sh"
