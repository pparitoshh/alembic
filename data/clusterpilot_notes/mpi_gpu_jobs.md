# Slurm: MPI and GPU job scripts (launch, binding, modules, profiling)

A typical job with one MPI rank per GPU on two nodes:

```bash
#!/bin/bash
#SBATCH --job-name=mpi-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4       # one MPI rank per GPU
#SBATCH --gpus-per-node=4         # some sites use --gres=gpu:4 instead
#SBATCH --cpus-per-task=8         # CPU cores per rank (for OpenMP threads)
#SBATCH --time=01:00:00

module purge
module load <compiler> <mpi> <cuda>   # names differ per site: find them with `module avail` or `module spider`
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpus-per-task=$SLURM_CPUS_PER_TASK --gpus-per-task=1 --gpu-bind=closest ./app
```

- Launch with `srun` inside the job: it starts the ranks on the allocated nodes with the right CPU and GPU binding.
  `mpirun` also works with MPI libraries built with Slurm support, but `srun` is the usual choice. `srun --mpi=list`
  shows the MPI plugins (for example `pmix`); use `srun --mpi=pmix` when the site's MPI needs it.
- Pass `--cpus-per-task` to `srun` as well: some Slurm versions do not pass the `#SBATCH --cpus-per-task` value on
  to `srun`, and every rank then gets a single core.
- Check the binding before a long run:
  `srun bash -c 'echo rank $SLURM_PROCID on $(hostname) GPU $CUDA_VISIBLE_DEVICES'`.
- Hybrid MPI + OpenMP: ranks x threads should equal the cores you asked for (`ntasks-per-node * cpus-per-task`).
- See GPU use while the job runs: `srun --jobid=<jobid> --overlap nvidia-smi`.
- Profile one rank per file with Nsight Systems: `srun nsys profile -o report_%q{SLURM_PROCID} ./app`;
  for kernel details use Nsight Compute (`ncu`) on a short run.
- Containers: `srun apptainer exec --nv image.sif ./app` (`--nv` makes the GPUs visible inside).
- Partition names, GPU types and module names are site specific: check `sinfo -o "%P %G"` and the site documentation.
