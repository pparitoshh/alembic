# srun, job steps and interactive jobs

Inside a batch script, `srun` launches a parallel job step using the resources of the allocation. For MPI programs built against Slurm's PMI support, `srun ./app` replaces `mpirun`. Each `srun` call creates a new job step, visible in `sacct` as `<jobid>.0`, `<jobid>.1`, and so on.

Options such as `--ntasks`, `--cpus-per-task` and `--gpus-per-task` can be given to `srun` to use a subset of the allocation for one step. Note that since Slurm 22.05, `srun` does not inherit `--cpus-per-task` from `sbatch`; set `SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK` or pass `--cpus-per-task` again.

Interactive work:

- `salloc --nodes=1 --time=01:00:00` obtains an allocation and starts a shell; run `srun` commands inside it and `exit` to release it.
- `srun --pty --time=00:30:00 --ntasks=1 bash` starts an interactive shell directly on a compute node.

Useful environment variables set in every job: `SLURM_JOB_ID`, `SLURM_JOB_NODELIST`, `SLURM_NTASKS`, `SLURM_CPUS_PER_TASK`, `SLURM_SUBMIT_DIR` (the directory `sbatch` was run from). By default a batch job starts in the submit directory; `--chdir=<dir>` (`-D`) changes it.
