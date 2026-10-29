# Multi-node GPU jobs that hang or are slow (NCCL)

If one node works and two nodes hang at the first collective (all_reduce, barrier) and then time out, the problem is
between the nodes. Check, in this order:

1. Rendezvous: every process uses the same address and port, the first host of
   `scontrol show hostnames "$SLURM_JOB_NODELIST"`, and that port is free.
2. Process count: with torchrun start one torchrun per node (`--ntasks-per-node=1`, `--nproc_per_node=<GPUs>`);
   `--ntasks-per-node=<GPUs>` together with torchrun starts too many processes.
3. Network: run once with `export NCCL_DEBUG=INFO` and look for `NET/IB` (InfiniBand) in the log. `NET/Socket` on a
   wrong interface can hang or be slow: set `NCCL_SOCKET_IFNAME` (and `GLOO_SOCKET_IFNAME`) to the cluster's
   interface, for example `ib0`.
4. Test the network alone with nccl-tests on the same nodes: `srun --ntasks-per-node=4 --gpus-per-task=1
   all_reduce_perf -b 8 -e 1G -f 2 -g 1`.

Raising the timeout (the `timeout` of `torch.distributed.init_process_group`) only delays the error. A reference
from our own measurement on two Leonardo Booster nodes: NCCL all_reduce of 1 GiB on 8 GPUs reached 84 GB/s over
InfiniBand and 4.5 GB/s with InfiniBand switched off (`NCCL_IB_DISABLE=1`); a result near the second number means
NCCL is not using InfiniBand.
