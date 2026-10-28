# Slurm: save a checkpoint and requeue a job before its time limit

`#SBATCH --requeue` alone does not restart a job that reaches its time limit: a job that runs out of time ends in
state TIMEOUT and stays finished. `--requeue` only allows the job to be put back in the queue (by `scontrol requeue`,
or by Slurm after a node failure or preemption). To continue after the time limit, the job must catch a warning
signal, save a checkpoint, and requeue itself:

```bash
#!/bin/bash
#SBATCH --job-name=long-run
#SBATCH --time=04:00:00
#SBATCH --requeue                 # allow the job to be requeued
#SBATCH --signal=B:USR1@300       # SIGUSR1 to the batch shell 300 s before the time limit
#SBATCH --open-mode=append        # keep one log across restarts
#SBATCH --output=long-run_%j.out

checkpoint_and_requeue() {
    echo "$(date): time limit near, checkpointing (restart $SLURM_RESTART_COUNT)"
    kill -USR1 "$PID"             # srun passes the signal to the program, which saves its state
    wait "$PID"
    scontrol requeue "$SLURM_JOB_ID"
}
trap checkpoint_and_requeue USR1

srun ./simulation --resume-from latest.ckpt &   # in the background, so the trap can run
PID=$!
wait "$PID"
```

Why each part matters:
- `B:` in `--signal=B:USR1@300` sends the signal to the batch shell, where the trap is. Without `B:` Slurm signals
  only the job steps (the `srun` tasks), and the script's trap never runs.
- Bash runs a trap only after the current foreground command ends. Starting `srun` with `&` and then `wait` lets the
  trap run as soon as the signal arrives.
- `scontrol requeue` puts the same job id back in the queue; `SLURM_RESTART_COUNT` counts the restarts, so the
  program can resume from its last checkpoint.
- `--open-mode=append` keeps the output of every run in the same file instead of overwriting it.
- Choose the warning time (`@300` seconds) longer than the time a checkpoint takes to write. Slurm may send the
  signal up to 60 seconds earlier than requested.

If the program writes checkpoints on its own (for example every 30 minutes), the trap only needs
`scontrol requeue "$SLURM_JOB_ID"`. Some sites also offer preemptible partitions or QOS that requeue jobs
automatically: check the site documentation.
