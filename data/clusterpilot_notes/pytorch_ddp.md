# Slurm: PyTorch distributed training (DDP) on several GPU nodes

Run one `torchrun` per node with `srun`; torchrun starts one process per GPU and gives each its own `RANK`,
`LOCAL_RANK` and `WORLD_SIZE`:

```bash
#!/bin/bash
#SBATCH --job-name=ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun per node
#SBATCH --gres=gpu:4                 # 4 GPUs per node
#SBATCH --cpus-per-task=32           # all cores of the node for the 4 processes
#SBATCH --time=02:00:00
#SBATCH --partition=boost_usr_prod   # Leonardo Booster; use your cluster's GPU partition elsewhere
#SBATCH --account=<project>          # Leonardo: the account name shown by `saldo -b`

module purge
module load cuda/12.2                # site specific: `module avail`
source $HOME/venvs/torch/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8             # cores per GPU process

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 \
     --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     train.py
```

In `train.py`: `torch.distributed.init_process_group("nccl")`, then
`local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)` and wrap the model in
`DistributedDataParallel(model, device_ids=[local_rank])`.

- Do not set `RANK` from `$SLURM_PROCID` in the batch script: the batch script runs once, so every process would
  get the same rank. torchrun (or `srun` with one task per GPU) sets the ranks.
- Without torchrun: `--ntasks-per-node=4 --gpus-per-task=1` and in Python read `SLURM_PROCID` (rank),
  `SLURM_LOCALID` (local rank) and `SLURM_NTASKS` (world size).
- `NCCL_DEBUG=INFO` prints which network NCCL uses; a slow run often means NCCL is not using InfiniBand.
