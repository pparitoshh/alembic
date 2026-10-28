"""All-reduce bus bandwidth with torch.distributed (NCCL) across the job's GPUs; sizes 1 MiB to 1 GiB, 20 timed iterations each.
Launched by commbench.sbatch with srun, one task per GPU (RANK/WORLD_SIZE from Slurm)."""
import os, time, torch, torch.distributed as dist
rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
torch.cuda.set_device(int(os.environ["SLURM_LOCALID"]))
dist.init_process_group("nccl", rank=rank, world_size=world)
for mib in (1, 4, 16, 64, 256, 1024):
    x = torch.ones(mib * 2**20 // 4, device="cuda")
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
    if rank == 0:
        busbw = mib * 2**20 * 2 * (world - 1) / world / dt / 1e9
        print(f"allreduce {mib:5d} MiB  time {dt*1e3:8.2f} ms  busbw {busbw:7.1f} GB/s  ranks {world}", flush=True)
dist.destroy_process_group()
