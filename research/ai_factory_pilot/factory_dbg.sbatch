#!/bin/bash
#SBATCH --job-name=dk-factory
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6  # 5 nodes x 24 + a 4-node score job (128) stay under the 256-CPU user limit
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# sbatch --nodes=N with PHASE=generate|review and SHARD_LIST="003 017 ..." (one shard per GPU).
# Bundled tail (Oct 26): with SHARD_LIST2 set, the job also runs another batch's leftover shards (BATCH2, PHASE2) on its
# last ceil(len(SHARD_LIST2)/4) nodes, as a second job step next to the first, so a 1-node tail does not take one of the
# user's two debug-QoS slots on its own.
L="${REPO:?}/../logs"
gpus() {  # $1 = phase, $2 = nodes
    # Review with REVIEW_TP=2: two tasks per node, two GPUs each (see factory_worker.sh); otherwise one GPU per task.
    if [ "$1" = review ] && [ "${REVIEW_TP:-1}" = 2 ]; then echo "--ntasks=$(( $2 * 2 )) --ntasks-per-node=2 --gpu-bind=none"  # each task picks its GPU pair; --ntasks overrides the allocation's 4 per node
    # Generate with GEN_TP=4: one task per node, all four GPUs (Qwen3-Next TP4 serving four shards).
    elif [ "$1" = generate ] && [ "${GEN_TP:-1}" = 4 ]; then echo "--ntasks=$2 --ntasks-per-node=1 --gpu-bind=none"
    else echo "--gpu-bind=single:1 --gpus-per-task=1 --ntasks=$(( $2 * 4 ))"; fi
}
if [ -z "${SHARD_LIST2:-}" ]; then
    srun --kill-on-bad-exit=0 $(gpus "${PHASE:?}" $SLURM_JOB_NUM_NODES | sed 's/ --ntasks=[0-9]*$//') \
        --output="$L/factory-${PHASE}-%j-%t.out" --error="$L/factory-${PHASE}-%j-%t.err" \
        bash "$REPO/research/ai_factory_pilot/factory_worker.sh"
    exit
fi
read -r -a HOSTS <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
N2=$(( ($(wc -w <<< "$SHARD_LIST2") + 3) / 4 )); N1=$(( ${#HOSTS[@]} - N2 ))
H1=$(IFS=,; echo "${HOSTS[*]:0:$N1}"); H2=$(IFS=,; echo "${HOSTS[*]:$N1}")
srun --kill-on-bad-exit=0 --nodes=$N1 --nodelist=$H1 $(gpus "${PHASE:?}" $N1) \
    --output="$L/factory-${PHASE}-%j-%t.out" --error="$L/factory-${PHASE}-%j-%t.err" \
    bash "$REPO/research/ai_factory_pilot/factory_worker.sh" &
BATCH=${BATCH2:?} PHASE=${PHASE2:?} SHARD_LIST=$SHARD_LIST2 \
    srun --kill-on-bad-exit=0 --nodes=$N2 --nodelist=$H2 $(gpus "$PHASE2" $N2) \
    --output="$L/factory-${PHASE2}-%j-b-%t.out" --error="$L/factory-${PHASE2}-%j-b-%t.err" \
    bash "$REPO/research/ai_factory_pilot/factory_worker.sh" &
wait
