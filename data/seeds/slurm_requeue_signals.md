# Requeue, signals and email notifications

Ask Slurm to email you with `--mail-type` and `--mail-user`: `#SBATCH --mail-type=END,FAIL` and `#SBATCH --mail-user=you@example.org`. Common types are `BEGIN`, `END`, `FAIL`, `REQUEUE` and `ALL`. `TIME_LIMIT_80` warns when 80% of the time limit has been used.

A job that is requeued goes back to the queue with the same job ID. Allow it with `#SBATCH --requeue` (or forbid it with `--no-requeue`). A running job can be requeued by hand with `scontrol requeue <jobid>`. By default a requeued job overwrites its output file; add `#SBATCH --open-mode=append` to keep the earlier output.

To checkpoint before the time limit ends, ask Slurm to send a signal ahead of time with `--signal`. `#SBATCH --signal=B:USR1@120` sends `SIGUSR1` to the batch shell 120 seconds before the limit. The `B:` prefix means the signal goes only to the batch script; without it the signal goes to all steps. Bash only runs a trap when the foreground command returns, so run the program in the background and `wait`:

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append

handler() {
  echo "checkpointing and requeueing"
  scontrol requeue $SLURM_JOB_ID
}
trap handler USR1

python train.py &
wait
```

Set a minimum acceptable run time with `--time-min`; Slurm may then shorten the time limit so the job can start earlier (backfill). Jobs killed for hitting the time limit end in state `TIMEOUT`; jobs killed for exceeding memory end in `OUT_OF_MEMORY`. Check the reason with `sacct -j <jobid> --format=JobID,State,ExitCode`.

Hold a pending job with `scontrol hold <jobid>` and let it run again with `scontrol release <jobid>`. Change a pending job's time limit with `scontrol update JobId=<jobid> TimeLimit=02:00:00`.
