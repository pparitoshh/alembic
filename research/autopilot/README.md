# Unattended HPC improvement pipeline

The coordinator finishes the remaining comparisons, runs new source-grounded RFT rounds, evaluates three training seeds, compares candidates with release 12e using both judges, checks commands and tool behavior, and exports a qualifying candidate. It stops after a candidate passes the release checks and its export is verified.

The objective incorporates the user's October 22 updates: **at least 90% on each of the three benchmarks under both judges, and first place on the internal leaderboard**. Beating the current release is an intermediate checkpoint. Scores measure judge preference against closed-book Qwen3-4B, not factual accuracy. These development results do not establish world-leading performance. The independent final holdout is excluded from training and is never evaluated by this loop.

## Release rule

A candidate must have complete, valid paired judgments on v2 (66 questions), v3 (200), and d1 (106). Its score difference from 12e must be positive on every slice under **both** GPT-OSS and Gemma. The existing v3 significance threshold must also pass under at least one judge. The tool and static-command checks must not regress. Judgments are checked against the exact saved answer bytes, and reported differences are recomputed from the raw verdicts.

It must also score at least 0.90 on **every slice under each judge**, and exceed every frozen competitor's unweighted three-slice mean under both judges. The reference includes all 13 configurations on the October 22 internal leaderboard. The candidate and leaderboard must share identical closed-book reference answers. Invalid competitor verdicts are conservatively assigned full credit when establishing the target; empty competitor answers remain in their original results. Judge self-preference remains a limitation, so both judging families are required. Reaching a mean of 90% while failing an individual slice does not pass.

Missing or invalid judgments do not become ties, and valid failed judgments are not retried away. Truncated answers remain in the results and are reported. These checks do not establish complete semantic correctness or live multi-turn tool competence. Repeated development-set selection is disclosed in the generated model card.

## Workflow

1. Verify the latest data release's digest and row count; resume its existing factory controller if needed.
2. Reconcile rounds `rft1` and `rag65kd11`, including their unfinished teacher and two-judge release comparisons.
3. Prepare an isolated RFT experiment. Use rotating thirds of the frozen, reviewed teacher bank and new reproducible sampling seeds. Alternate training samples with a concise-completion instruction and samples using the unchanged evaluation prompt. This tests a response to the high truncation rate in prior rounds without changing evaluation. Generate four candidates per eligible question, retain the existing 0.75 teacher-comparison threshold, and require at least 500 accepted training rows.
4. Train seeds 42, 43 and 44, evaluate the existing strengths, and retain all results. Every round starts from the frozen release 12e, so an unsuccessful intermediate model does not silently become the baseline.
5. Compare with the teacher and release 12e. If the promotion rule fails, automatically prepare the next round. Thresholds, evaluation inputs, and the baseline remain fixed.
6. For a benchmark winner, run first-response tool and static-command non-regression checks. If these fail, continue research.
7. Merge, export, evaluate Q8_0 and Q4_K_M GGUF artifacts, write a model card and checksums, and stage `runs/hf_<round>/READY`. Mark the objective complete only after verifying the staged files and quantized evaluation coverage.

Public upload is separate: the service holds no publishing credentials and does not overwrite the published model. Laptop RAM, latency and generation speed require measurements on the deployment laptop; GPU evaluation is not represented as a laptop benchmark.

## Durability and resources

The coordinator runs in a one-CPU, 2 GiB Slurm allocation on `lrd_all_serial`. It prequeues a chain of three successors with `afterany` dependencies and hands over before the four-hour CPU partition limit. The extra links survive a successor's failure in the cluster startup procedure, before Python can run. A healthy successor replenishes the chain. This survives disconnection, coordinator death, isolated startup failures and normal allocation expiry without requiring a Mac or user cron. Slurm outages still delay work, and repeated startup failures can exhaust the queued chain.

Long-running controller scripts use independent four-CPU, 28 GiB allocations. Together with the coordinator, this uses 30,720 MiB, below the serial partition's 30,800 MiB per-user limit. Merging and conversion run within that allocation, without requesting a nested CPU job. GPU jobs keep the existing `boost_qos_dbg` **30-minute** limit.

Two coordinated queues share eight nodes (32 A100s) and at most two GPU job slots. The foreground queue trains and evaluates; the background queue generates and judges data for up to six future experiments. Typical splits are 3+5 nodes for training, 2+6 for answering, and 4+4 for judging. Independent judging workers can shrink to fit the available share; training and answer layouts remain fixed. If the foreground is waiting for its own training data, preparation can use all eight nodes. Live allocations cannot grow mid-job, so phase transitions, Slurm scheduling and model loading can leave temporary gaps. Allocation is reported separately from actual GPU utilization.

Generation uses stable 64-question units, partial journals and atomic completion files. A subsequent job can use a different node count without duplicating questions. Judgments retain invalid verdicts and bind to hashed samples. Only datasets with at least 500 rows at the unchanged 0.75 threshold become ready for training. Insufficient datasets are recorded as rejected and the loop advances. Three allocations without progress stop for inspection. GPU pool leases reserve nodes even when a submission reply or accounting result is uncertain. Existing jobs, including jobs outside the pool, count against the same limit. No other user's jobs are cancelled.

If training is waiting for its own samples and a smaller sampling allocation is already running, a second sampling job uses the free nodes. Unit locks prevent duplicate writers, and judging waits for both allocations to finish. This assistance applies only to generation, whose units are locked; it never duplicates judge calls.

If the complete sample audit passes before the optional helper finishes, the coordinator records and cancels only that redundant helper. A queued helper therefore cannot delay judging after its work is already done. Manual cancellations still stop automatic resubmission.

The account currently exposes an eight-node debug allowance. No reservation for this account was visible at deployment preflight; a dedicated reservation must be identified before it can be targeted. The actual serial preflight started within two seconds despite a much later `sbatch --test-only` forecast.

Every submission first writes an intent receipt. A lost `sbatch` response is reconciled with `squeue` and `sacct`; it is never blindly resubmitted. Failed scheduler reads never count as an empty queue. Timeouts and node failures resume from existing artifacts; three genuine task failures stop the service for inspection. Manual cancellation is respected. A filesystem lock prevents competing coordinators. Legacy `DONE` log messages are informational, never proof of completion.

Round controllers wait for a terminal state in `sacct`, so a finished job being removed from the live `squeue` cache cannot leave a controller waiting indefinitely. Accounting delays and errors continue waiting, while cancellation stops the controller. An elastic foreground judge uses the remaining node capacity when the other GPU slot is already occupied; the four-node default applies only when starting in an empty pool.

## Deployment

Copy this directory to `<campaign>/ctl/autopilot/runtime`, retaining `templates/`. The templates are snapshots of the previously running controllers, with provenance in `templates/SOURCES.json`.

```bash
python runtime/configure.py --root /absolute/campaign --python /absolute/runtime/bin/python
python runtime/supervisor.py bootstrap /absolute/campaign/ctl/autopilot/config.json
```

Configuration hashes the existing model, teacher bank, baseline answers, evaluation files and relevant code. It renders private controller copies without changing historical experiments or their logs. Existing files must match on repeated preparation. Configuration does not submit any jobs.

Before starting, use `sbatch --test-only` to check the CPU resource request, run the read-only evidence audits, and syntax-check the rendered shell scripts. The local regression suites are `tests/test_autopilot.py` and `tests/test_autopilot_pool.py`. Existing installations can use `enable_pool.py CONFIG` before any automatic round has been prepared with the old worker layout. Upgrade the queued coordinator to 2 GiB and roll over that coordinator; its independent GPU jobs continue.

For the accounting-wait repair, deploy `wait_gpu.py`, `upgrade_waiter.py` and the revised preparation/pool code. Pause new submissions, run `upgrade_waiter.py CONFIG`, and restart any controller still executing the old wait function. The upgrader verifies each prepared controller's digest, backs up its script and plan, changes only the wait function, and updates the recorded controller digest. Saved data, answers, judgments and experiment settings remain intact. It is safe to resume an interrupted upgrade. Regression coverage is in `tests/test_autopilot_wait.py`.

## Status and controls

Read `<campaign>/ctl/autopilot/STATUS.md` for the current phase and job IDs. `STATE.json`, `decisions/`, `submissions/`, `gpu/` and `logs/` retain the detailed evidence. Each new candidate has an `AUTOPILOT_PLAN.json` with its data, sampling seed and controller digest.

The coordinator writes `reports/LATEST.md` and timestamped JSON/Markdown reports every 30 minutes. Reports include active jobs, completed decisions, the published score table, the best minimum of the six scores, and its change since the previous report. An incomplete evaluation never becomes a reported improvement. `mirror_reports.py` can be run every 1,800 seconds by a local LaunchAgent to mirror these files into the project. Remote work and remote reports continue without the Mac; local copies refresh while it is awake and connected. `SYNC.json` records the last successful refresh and any connection failure. Chat notifications are not configured.

```bash
python runtime/supervisor.py status /absolute/campaign/ctl/autopilot/config.json
touch /absolute/campaign/ctl/autopilot/PAUSE    # no new managed GPU submissions; running jobs finish
rm /absolute/campaign/ctl/autopilot/PAUSE       # resume
touch /absolute/campaign/ctl/autopilot/STOP     # stop coordinating, preserve existing GPU jobs
```

A blocked or stopped service needs inspection and an explicit state reset before restarting. Do not start the legacy watchdog alongside this coordinator. A passing result is not guaranteed; the loop can continue producing new candidates until the objective is met or a recorded failure requires intervention.
