# Slurm eval: hf.co/VaibhavKKM/hpc-assistant-qwen3-4b:latest

Date: 2026-10-25. Cluster: container `slurm-test` (ubuntu:24.04, slurm-wlm 23.11.4), user `ubuntu`, node `slurmnode` (12 CPUs, 32000 MB, gpu:4), partitions `gpu` + `boost_usr_prod`, accounting via slurmdbd. Prompts and expected answers were written before the model calls (see `prompts.txt`, `expected.md`). The model answers are in `results.json` and `results.md`. The real command output for each prompt is in `results.json` and `results.md`.

## Score

| Rating | Count | Items |
|---|---|---|
| correct | 6 | 1, 3, 4, 5, 7, 8 |
| partly correct | 2 | 6, 10 |
| wrong | 2 | 2, 9 |

## Results table

| # | Task | Model answer | Real output | Rating / reason |
|---|---|---|---|---|
| 1 | sbatch header (1 node, 8 CPU, 16G, 2h, name, output with %j) | full header with `--cpus-per-task=8 --mem=16G --time=02:00:00 --job-name=myjob --output=myjob_%j.out` | job 65 ran: `ReqTRES=cpu=8,mem=16G`, `TimeLimit=02:00:00`, file `myjob_65.out` contains `RUN-OK` | **correct** - all flags right, `%j` works |
| 2 | array 1-10, max 3 concurrent | `#SBATCH --array=1-10:3` | accepted, but sacct shows only tasks `66_1, 66_4, 66_7, 66_10` | **wrong** - colon is the step syntax (4 tasks, no throttle); must be `--array=1-10%3` |
| 3 | dependency afterok | `sbatch --dependency=afterok:42 B.sh` | `Submitted batch job 71`; 71 COMPLETED after real job 42 | **correct** - exact expected form, ran |
| 4 | mail on end/fail | `--mail-user=me@example.com` + `--mail-type=END,FAIL` | job 70 accepted and completed; delivery fails only because `/bin/mail` (MailProg) is missing in the container | **correct** - flags right; delivery error is environmental |
| 5 | srun 4 tasks | `srun -n 4 hostname` | 4 lines `slurmnode`, RC=0 | **correct** |
| 6 | sacct elapsed+state for job 99 | `sacct -j 99 --format=Elapsed,State` | empty table (job 99 does not exist); accounting IS active (slurmdbd), so no error; bonus with real job: `00:00:00 COMPLETED` | **partly** - works but omits the JobID column |
| 7 | scontrol show job 77 | `scontrol show job 77` (+ squeue/sacct pre-checks) | `slurm_load_jobs error: Invalid job id specified` (77 does not exist); bonus on real job 70 works. Side note: its sacct line uses invalid field `EndTime` | **correct** - main command exactly right |
| 8 | sinfo 3 columns | `sinfo -o "%P %N %T"` | `PARTITION NODELIST STATE`, one line per node, both partitions | **correct** |
| 9 | hold/release job 55 | `scancel --suspend 55` / `scontrol resume 55` | `scancel: unrecognized option '--suspend'`; `scontrol resume 55` -> `Access/permission denied for job 55`; same on a real running job | **wrong** - scancel has no --suspend; expected `scontrol hold`/`release` |
| 10 | module + srun python | script with `#SBATCH --partition=compute`, `module load python`, `srun python -c "print(42)"` | `sbatch: error: invalid partition specified: compute` (RC=1); container also has no `module` command | **partly** - structure right, partition name hallucinated |

## Environment notes

- The node was declared with 12 CPUs and 32000 MB so the 8-CPU/16G header could actually run; declared memory larger than reported needed `SlurmctldParameters=node_reg_mem_percent=40` to avoid auto-drain.
- The container HAS accounting (slurmdbd + MariaDB), so item 6 produced an empty result, not an "accounting disabled" error.
- Only file names and the B.sh shebang were adapted when running model answers; no model errors were fixed.

## Accuracy

Computed by `../../accuracy.py` (standard library only). Round 1 ratings come
from `round1_smoke.md`; round 2 ratings come from `results.json` in this
folder.

| model / round | n | strict | lenient | Wilson 95% (strict) | format |
|---|---|---|---|---|---|
| round 2, qwen3-4b | 10 | 60.0% | 70.0% | [31.3%, 83.2%] | n/a |
| round 1, qwen3-4b | 4 | 75.0% | 75.0% | [30.1%, 95.4%] | 4/4 |
| round 1, hpc-assistant-x02 | 4 | 25.0% | 25.0% | [4.6%, 69.9%] | 0/4 |

Strict accuracy is the share of fully correct answers. Lenient accuracy counts
a partly correct answer as half a point. The Wilson 95% interval shows the
range where the true strict accuracy most likely sits. Format-follow rate is
the share of answers that obey the prompt's format instruction; round 2 had no
format instruction, so the rate is n/a.

The sample is small (4 and 10 prompts, one run each), so the intervals are
wide and the difference between the models is not proven.

### Round 2 per item (each item is one task type)

| id | task | rating |
|---|---|---|
| 1 | sbatch header: 1 node, 8 CPUs, 16G, 2h, job name, output with job id | correct |
| 2 | job array 1-10, max 3 concurrent | wrong |
| 3 | dependency: B after A (42) ends OK | correct |
| 4 | mail on end/failure | correct |
| 5 | srun with 4 tasks | correct |
| 6 | sacct elapsed and state | partly correct |
| 7 | scontrol show job | correct |
| 8 | sinfo custom format: partition, node, state | correct |
| 9 | hold and release job 55 | wrong |
| 10 | script: module load python + srun python -c 'print(42)' | partly correct |
