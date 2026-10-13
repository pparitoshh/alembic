#!/bin/bash
#SBATCH --job-name=dk-check-fanout
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=slurm/logs/%x-%j.out
# Checks the whole-node seed method (fan_out_seeds) before a real run: each seed task prints its seed,
# GPU, cores and port. Expect one line per seed, each with a different GPU and port.
source "${REPO:-$WORK/alembic}/slurm/env.sh"
source "$REPO/slurm/common.sh"
fan_out_seeds slurm/check_fanout.sbatch   # one GPU task per seed; below runs once per seed

echo "[check] host=$(hostname) task=$(task_index) seed=$(seed_of_task) port=$(job_port 1)" \
     "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} cores=$(nproc)" \
     "gpu=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr '\n' ' ')"
"$VENV/bin/python" -c "import torch; print('[check] torch sees', torch.cuda.device_count(), 'GPU:', torch.cuda.get_device_name(0))"
