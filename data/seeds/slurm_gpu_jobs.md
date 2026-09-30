# Requesting GPUs in Slurm

GPUs are generic resources (GRES). Request them with `--gres=gpu:<count>` or `--gres=gpu:<type>:<count>`, for example `--gres=gpu:a100:2`. Newer Slurm versions also offer `--gpus=<n>` (total GPUs for the job), `--gpus-per-node=<n>`, `--gpus-per-task=<n>`, and `--cpus-per-gpu=<n>`.

Slurm sets `CUDA_VISIBLE_DEVICES` for the job so that the program only sees its allocated GPUs. Do not overwrite this variable manually.

Example: one node, 4 GPUs, one task per GPU:

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

srun python train.py
```

For PyTorch multi-GPU training on one node, a common pattern is a single task that launches its own workers: `--ntasks=1`, `--gpus-per-node=4`, then `torchrun --nproc_per_node=4 train.py`.

Use `--constraint=<feature>` (`-C`) to select nodes with a specific feature defined by the admins, e.g. a GPU model. Use `--exclusive` to get whole nodes without sharing them with other jobs.

Check GPU usage inside a job with `nvidia-smi`, or attach to a running job with `srun --jobid=<jobid> --overlap --pty nvidia-smi`.
