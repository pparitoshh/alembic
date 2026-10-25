#!/bin/bash
#SBATCH --job-name=dk-rag65-probe
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
# Node 0: Qwen3-235B-A22B-Instruct-2507 (AWQ, TP4), rag + closed. Node 1: Qwen3-32B-AWQ with thinking (GPUs 0-1)
# and without (GPUs 2-3), rag. Named dk-rag65-* so the factory controllers yield the slot. Resubmit to resume.
: "${REPO:?}"
L="$REPO/../logs"
srun --kill-on-bad-exit=0 bash -c 'bash "$REPO/research/teacher_probe/probe_worker.sh"' > "$L/probe-$SLURM_JOB_ID.out" 2>&1
