# Leonardo (CINECA): partitions, QOS, budget, storage and modules

Booster (GPU) nodes: 4 NVIDIA A100 64 GB GPUs and 32 cores each. Ask for GPUs with `--gres=gpu:N` (N up to 4 per node).

| Partition | QOS | Limits per job | Wall time |
|---|---|---|---|
| `boost_usr_prod` | `normal` (default) | up to 64 nodes | 24 h |
| `boost_usr_prod` | `boost_qos_dbg` | up to 8 nodes; 2 jobs per user | 30 min |
| `boost_usr_prod` | `boost_qos_bprod` | 65 to 256 full nodes | 24 h |
| `boost_usr_prod` | `boost_qos_lprod` | up to 8 nodes | 4 days |
| `lrd_all_serial` (default partition) | `normal` | up to 4 cores, no GPUs, free of budget | 4 h |
| `dcgp_usr_prod` (CPU nodes) | `normal` | up to 16 nodes | 24 h |

- A GPU job needs `#SBATCH --partition=boost_usr_prod` and `#SBATCH --account=<project>`; for a short test add
  `#SBATCH --qos=boost_qos_dbg`. QOS names that start with `qos_fua` are only for EUROfusion projects.
- Budget: `saldo -b` lists your project accounts (use this name in `--account`), the budget in local core-hours
  and how much is used. When a project's budget is used up, sbatch fails with "invalid account or expired budget".
- QOS limits: `sacctmgr show qos boost_qos_dbg format=name,maxwall,maxtrespu` (there is no `scontrol show qos`).
- Storage: `$HOME` (50 GB, backed up), `$WORK` (project, 1 TB by default), `$FAST` (project, faster I/O, 1 TB),
  `$CINECA_SCRATCH` (temporary: files older than 40 days are deleted), `$TMPDIR` (removed at the end of the job).
  Check usage and quotas with `cindata` (or `cinQuota`).
- Modules: `module avail`; for example `cuda/12.2`, `nccl`, `hpcx-mpi/2.19`, `openmpi/4.1.6--gcc--12.2.0-cuda-12.2`,
  `python/3.11.7`. Load them in the job script after `module purge`.
