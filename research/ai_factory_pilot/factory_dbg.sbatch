#!/bin/bash
#SBATCH --job-name=dk-factory
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# sbatch --nodes=N with PHASE=generate|review and SHARD_LIST="003 017 ..." (one shard per GPU).
L="${REPO:?}/../logs"
# Review with REVIEW_TP=2: two tasks per node, two GPUs each (see factory_worker.sh); otherwise one GPU per task.
if [ "$PHASE" = review ] && [ "${REVIEW_TP:-1}" = 2 ]; then GPUS="--ntasks=$(( SLURM_JOB_NUM_NODES * 2 )) --ntasks-per-node=2 --gpu-bind=none"  # each task picks its GPU pair; --ntasks overrides the allocation's 4 per node
else GPUS="--gpu-bind=single:1 --gpus-per-task=1"; fi
srun --kill-on-bad-exit=0 $GPUS \
    --output="$L/factory-${PHASE:?}-%j-%t.out" --error="$L/factory-${PHASE}-%j-%t.err" \
    bash "$REPO/research/ai_factory_pilot/factory_worker.sh"
