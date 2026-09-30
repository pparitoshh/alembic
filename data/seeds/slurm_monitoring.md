# Monitoring and managing jobs

- `squeue -u $USER` lists your pending and running jobs. The `ST` column shows the state: `PD` pending, `R` running, `CG` completing. The `NODELIST(REASON)` column explains why a pending job waits, e.g. `Priority`, `Resources`, `QOSMaxJobsPerUserLimit`, or `ReqNodeNotAvail`.
- `squeue --start -j <jobid>` shows the scheduler's estimated start time.
- `scancel <jobid>` cancels a job. `scancel -u $USER` cancels all your jobs. `scancel --name=<name>` cancels jobs by name.
- `scontrol show job <jobid>` prints full details of a job, including the allocated nodes and the working directory.
- `scontrol hold <jobid>` / `scontrol release <jobid>` hold and release a pending job.
- `sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode` shows accounting data for finished jobs. `MaxRSS` is the peak memory used by a job step, useful for tuning `--mem`.
- `seff <jobid>` (if installed) summarises CPU and memory efficiency of a finished job.
- `sinfo` shows partitions and node states (`idle`, `mix`, `alloc`, `drain`, `down`).

A job that ran out of memory usually ends with state `OUT_OF_MEMORY`; one that hit its time limit ends with `TIMEOUT`. Increase `--mem` or `--time` accordingly.
