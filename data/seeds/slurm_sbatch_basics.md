# Submitting batch jobs with sbatch

`sbatch` submits a batch script to Slurm. The script is a shell script whose first line is a shebang such as `#!/bin/bash`. Lines that begin with `#SBATCH` before the first executable command are parsed as options; options given on the command line override them.

Common resource options:

- `--job-name=<name>` (`-J`): name shown in `squeue`.
- `--time=<time>` (`-t`): wall-clock limit, e.g. `--time=02:00:00` or `--time=1-12:00:00` (days-hours:minutes:seconds). A job that exceeds its limit is killed.
- `--nodes=<n>` (`-N`): number of nodes.
- `--ntasks=<n>` (`-n`): number of tasks (processes, typically MPI ranks).
- `--ntasks-per-node=<n>`: tasks per node.
- `--cpus-per-task=<n>` (`-c`): CPU cores per task, used for threaded (OpenMP) programs. Set `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` in the script.
- `--mem=<size>`: memory per node, e.g. `--mem=16G`. `--mem-per-cpu=<size>` sets memory per allocated CPU instead; the two are mutually exclusive.
- `--partition=<name>` (`-p`): partition (queue) to submit to.
- `--account=<name>` (`-A`): account to charge.
- `--output=<file>` (`-o`) and `--error=<file>` (`-e`): paths for stdout and stderr. `%j` expands to the job ID and `%x` to the job name, e.g. `--output=%x-%j.out`.

Example:

```bash
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./my_program
```

Submit with `sbatch job.sh`. `sbatch` prints `Submitted batch job <jobid>`. The option `--test-only` validates the script and estimates when the job would start without actually submitting it.
