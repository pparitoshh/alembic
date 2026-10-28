# Slurm job arrays: many similar tasks in one submission

A job array runs the same script many times with a different index each time:

```bash
#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --array=0-99%10           # 100 tasks, at most 10 running at the same time
#SBATCH --time=00:30:00
#SBATCH --output=sweep_%A_%a.out  # %A = array job id, %a = task index

python train.py --config configs/run_${SLURM_ARRAY_TASK_ID}.yaml
```

- `SLURM_ARRAY_TASK_ID` is the index of this task; `SLURM_ARRAY_JOB_ID` is the id shared by all tasks;
  `SLURM_ARRAY_TASK_COUNT`, `SLURM_ARRAY_TASK_MIN` and `SLURM_ARRAY_TASK_MAX` describe the whole array.
- Index lists and steps: `--array=1,5,7`, `--array=1-99:2` (odd numbers only).
- `%10` limits how many tasks run at once. Change it later with `scontrol update JobId=<array_job_id> ArrayTaskThrottle=5`.
- One line per task in a file: `LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" inputs.txt)`.
- Each task is its own job `<array_job_id>_<index>`: `squeue -r` lists them one per line;
  `scancel 1234_7` cancels one task, `scancel 1234` the whole array; `sacct -j 1234` shows every task.
- Rerun only the failed tasks: list them with `sacct -j 1234 --state=FAILED,TIMEOUT -X -n -o JobID` and submit the
  same script with `--array=` set to those indexes.
- Dependencies: `--dependency=afterok:1234` waits until every task of array 1234 succeeded;
  `--dependency=aftercorr:1234` starts task N of a second array when task N of the first succeeded;
  `--dependency=afterany:1234` waits until all tasks ended, in any state.
- Sites limit the array size (`MaxArraySize`) and the number of jobs per user: check `scontrol show config | grep -i array`.
