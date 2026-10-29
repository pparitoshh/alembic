#!/bin/bash
#SBATCH --job-name=cp-mt
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
# Fix 4 pipeline as one 4-GPU job: collect.sh -> build.sh -> train.sh (+ eval.sh) for one SEED.
# All writes go to WORK (default ~/mt). The Slurm output goes to the submit directory, or give
# --output=$HOME/mt/logs/mt-%j.out on the sbatch command. Submit it with --exclude=<agent node>.
# usage: sbatch --exclude=NODE --export=ALL,SEED=42[,NAME=..,TEACHER=..,MODE=..,SMOKE=1] pipeline.sbatch
export NGPU=4 SEED=${SEED:-42} WORK=${WORK:-$HOME/mt}
D=${REPO:-$WORK/alembic}/research/train_multistep_tools/curiosity
mkdir -p $WORK/logs
L=$WORK/logs/mt_${NAME:-v1}$([ "${SMOKE:-0}" = 1 ] && echo _smoke)_s$SEED.log
echo "$(date -u +%FT%TZ) mt $SLURM_JOB_ID on $(hostname): start seed $SEED" >> $WORK/logs/tasks.log
rc=0
for s in collect build train; do
    bash $D/$s.sh >> $L 2>&1 || { rc=$?; echo "$(date -u +%FT%TZ) mt: $s failed rc=$rc" >> $L; break; }
done
echo "$(date -u +%FT%TZ) mt $SLURM_JOB_ID: end seed $SEED rc=$rc" >> $WORK/logs/tasks.log
exit $rc
