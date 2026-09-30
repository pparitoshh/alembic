# Job arrays and dependencies

A job array submits many similar jobs with one script: `--array=0-99` creates 100 tasks. Each task gets `SLURM_ARRAY_TASK_ID` (its index) and `SLURM_ARRAY_JOB_ID` (the shared array ID). Ranges can have a step (`--array=0-20:5`) or a list (`--array=1,3,7`). A `%` suffix limits concurrency: `--array=0-999%50` runs at most 50 tasks at the same time.

In output file names, `%A` expands to the array job ID and `%a` to the task index, e.g. `--output=logs/%x_%A_%a.out`.

Example processing one input file per task:

```bash
#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --array=0-9
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%A_%a.out

FILES=(data/*.csv)
python preprocess.py "${FILES[$SLURM_ARRAY_TASK_ID]}"
```

Cancel a single task with `scancel <arrayjobid>_<index>`, or the whole array with `scancel <arrayjobid>`.

Dependencies chain jobs: `sbatch --dependency=afterok:<jobid> next.sh` starts `next.sh` only after the given job completed successfully. Other types are `afterany` (after it ends in any state), `afternotok` (only if it failed) and `singleton` (only one job with this name and user runs at a time). Capture a job ID in a script with `jid=$(sbatch --parsable first.sh)`.
