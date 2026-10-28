# Node-to-node communication on Leonardo Booster: measurements (Oct 27, 2026)

Measured by Team Alembic on two Leonardo Booster nodes (lrdn0561, lrdn0697) inside the hackathon reservation, with
`commbench.sbatch` in this folder. Each node has 4 NVIDIA A100 64 GB GPUs and 4 InfiniBand HDR100 adapters
(Mellanox ConnectX-6, `mlx5_0` to `mlx5_3`, 100 Gb/s each). Software: OSU Micro-Benchmarks 7.4 built with CUDA-aware
Open MPI 4.1.6 (`osu-micro-benchmarks/7.4--openmpi--4.1.6--gcc--12.2.0-cuda-12.2` module), and PyTorch with its bundled
NCCL for the all-reduce tests. One run per setting, so treat differences below about 5% as noise.

## MPI point-to-point (OSU, 4 MiB messages)

| Test | Result |
|---|---|
| `osu_bw` between nodes, host memory | 24.7 GB/s |
| `osu_bw -d cuda D D` between nodes, GPU memory | 12.3 GB/s |
| `osu_bibw -d cuda D D` between nodes, GPU memory | 24.6 GB/s |
| `osu_bw -d cuda D D` two GPUs in the same node (NVLink) | 87.1 GB/s |
| `osu_latency` between nodes, host memory, 1 byte | 1.37 µs |
| `osu_latency -d cuda D D` between nodes, GPU memory, 1 byte | 1.99 µs |

One MPI rank per node uses one InfiniBand adapter, so a single pair of ranks reaches about one HDR100 link's worth of
host-to-host bandwidth. With GPU buffers, the default Open MPI/UCX settings on this module reached about half of that
for a single pair; inside a node, NVLink is about seven times faster than the network.

## MPI collective (OSU, 8 ranks = 4 GPUs on each of 2 nodes)

| Test | 256 MiB average latency |
|---|---|
| `osu_allreduce -d cuda` (GPU buffers) | 772 ms |
| `osu_allreduce` (host buffers) | 199 ms |

CUDA-aware MPI all-reduce on GPU buffers was about 4 times slower than on host buffers here. For collectives on GPU data
in deep-learning jobs, use NCCL (below) instead of MPI collectives on GPU buffers.

## NCCL all-reduce (8 GPUs on 2 nodes, 1 GiB, bus bandwidth)

| NCCL setting | Bus bandwidth | Compared with default |
|---|---|---|
| default (same speed as Tree) | 84.3 GB/s | |
| `NCCL_ALGO=Tree` | 84.4 GB/s | same |
| `NCCL_ALGO=Ring` | 48.7 GB/s | 42% lower |
| `NCCL_NET_GDR_LEVEL=LOC` (GPUDirect RDMA off) | 39.2 GB/s | 2.1 times lower |
| `NCCL_IB_DISABLE=1` (InfiniBand off, TCP sockets) | 4.5 GB/s | 19 times lower |

With `NCCL_DEBUG=INFO`, NCCL reported `NET/IB` with the four `mlx5` adapters at 100000 Mb/s each, so it used InfiniBand
and GPUDirect RDMA by default.

## Practical advice for multi-node GPU jobs on Leonardo

- Keep NCCL on InfiniBand: if a job's log shows `NET/Socket` instead of `NET/IB`, all-reduce can be about 19 times slower.
  Check with `NCCL_DEBUG=INFO` and look for `NET/IB` and the `mlx5` devices.
- Leave GPUDirect RDMA on (do not lower `NCCL_NET_GDR_LEVEL`); turning it off halved all-reduce bandwidth.
- Do not force `NCCL_ALGO=Ring` for large all-reduces across nodes; the default was 1.7 times faster.
- Use NCCL, not CUDA-aware MPI collectives, for all-reduce on GPU tensors.
- Within a node, GPU-to-GPU traffic over NVLink (87 GB/s) is far faster than the network, so place tightly coupled
  ranks on the same node where possible.
