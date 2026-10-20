#!/bin/bash
#SBATCH --job-name=dk-bestofn
#SBATCH --account=EUHPC_D30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclude=lrdn0812,lrdn1020,lrdn2071,lrdn2747,lrdn3050,lrdn3075
# sbatch --nodes=N --export=ALL,PHASE=cands|select,TEACHER=tnext|t32,NSHARDS=40 with SHARD_LIST in the env.
L="${REPO:?}/../logs"
if [ "$PHASE" = select ]; then GEO="--ntasks=$(( SLURM_JOB_NUM_NODES * 2 )) --ntasks-per-node=2 --gpu-bind=none"
elif [ "$TEACHER" = tnext ]; then GEO="--ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpu-bind=none"
else GEO="--gpu-bind=single:1 --gpus-per-task=1"; fi
srun --kill-on-bad-exit=0 $GEO --output="$L/bestofn-${PHASE}-%j-%t.out" --error="$L/bestofn-${PHASE}-%j-%t.err" \
    bash "$REPO/research/bestofn/worker.sh"
