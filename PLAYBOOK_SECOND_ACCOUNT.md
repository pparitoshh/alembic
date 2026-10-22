# Playbook: second Leonardo account (parallel training rounds)

This file is for a Claude Code session working on a **second member account of the Leonardo project `EUHPC_D30_031`**. The main pipeline runs on account `a08trc09` and is managed by another Claude Code session. Your job is to run extra training rounds with **your account's own GPU quota**, so the project gets twice the compute. Read the whole file before running anything.

## 0. Ground rules (non-negotiable)

1. **Only the account holder's own login is used.** CINECA accounts are personal: never use another person's credentials or SSH keys, and never ask for them.
2. **Write only inside `/leonardo_work/EUHPC_D30_031/shared_split/`.** Everything under `/leonardo_work/EUHPC_D30_031/runs/a08trc09/` is the main pipeline: read it, never modify, move or delete anything there.
3. **Never `scancel` a job you did not submit**, and do not cancel your own running jobs unless this file says so. Jobs end on their own (30-minute limit).
4. **`pkill` / `pgrep` patterns must be anchored**, e.g. `pkill -f '^bash /leonardo_work/EUHPC_D30_031/shared_split/ctl/run_round.sh sh24'`. An unanchored pattern can kill your own SSH session.
5. Do not upload models or data anywhere (no Hugging Face, no git pushes of data). Releases are done by the main session.
6. Report to the user after every check (what is running, progress, problems fixed).

## 1. The system

| Item | Value |
|---|---|
| Cluster | Leonardo Booster (CINECA). One node = 4 × NVIDIA A100 64 GB, 32 CPU cores, ~512 GB RAM |
| Project / account | `EUHPC_D30_031` (shared GPU budget, about 50,000 node hours left on Oct 22) |
| Queue we use | partition `boost_usr_prod`, QoS `boost_qos_dbg` (already set in every sbatch file) |
| **Your limits (per user)** | **2 running jobs, 8 nodes, 256 CPUs, ~3.86 TB memory, 4 submitted jobs (running + pending)**, 30 min per job |
| Normal QoS | unusable this month: jobs would start around Oct 30 (the project used more than its monthly share) |
| Home quota | **50 GB**. vLLM/Triton caches must not live in home: `run_round.sh` already points them to `shared_split/cache/$USER` |
| Access ends | ~Oct 28, 2026. All GPU work must be finished by Oct 26 evening |
| Login nodes | `login01`…`login07`. **Background processes live on the login node where you start them**; `pgrep` only sees processes on the node you are on. Write down the node (`hostname`) when you start something |

Jobs are resumable: every controller re-attaches to its own running job and continues from checkpoints after a 30-minute job ends. A `TIMEOUT` state is normal for training pieces.

## 2. One-time checks (5 minutes)

```bash
saldo -b | grep EUHPC_D30_031          # must print the project line
id -Gn | tr ' ' '\n' | grep EUHPC_D30_031   # must print EUHPC_D30_031
S=/leonardo_work/EUHPC_D30_031/shared_split
ls $S/ctl $S/code                       # must list run_round.sh, setup_env.sh / code folders
tail -3 $S/ctl/setup_env.out            # must end with SETUP_DONE (the main session prepared the Python copies)
$S/venv-train/bin/python -c "import torch, transformers; print('ok', torch.__version__)"
ls /leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code/runs/gguf_op235/merged/model.safetensors
```

If any check fails, stop and report the exact error to the user (the main session fixes permissions on its side).

## 3. Your task now: rounds 24 and 25

Both distil **Qwen3-235B** (the strongest teacher, same tokenizer as our Qwen3-4B student) into the **current release (round 20)**, using round 21's teacher scores, which the main pipeline is computing now:

| Round | Recipe | Nodes |
|---|---|---|
| `sh24` | distillation weight 0.8, LoRA r32/a64 (the recipe of our best jump so far) | 3 (one seed per node), then 1 for answers |
| `sh25` | distillation weight 1.0, LoRA r64/a128 (stronger teacher signal) | 3, then 1 |

Together they use 6 of your 8 nodes and both job slots. Start both on the **same login node**:

```bash
S=/leonardo_work/EUHPC_D30_031/shared_split
hostname | tee $S/ctl/LOGIN_NODE_$USER
for rd in sh24 sh25; do
  (setsid nohup bash $S/ctl/run_round.sh $rd >> $S/ctl/$rd/nohup.out 2>&1 < /dev/null &)
done
sleep 5; pgrep -af "^bash $S/ctl/run_round.sh"
tail -3 $S/ctl/sh24/controller.log $S/ctl/sh25/controller.log
```

What happens (all automatic):
1. `waiting for round 21's Qwen3-235B scores` — normal until the main pipeline logs `SCORED` (expected ~22:00 CEST Oct 22).
2. Training: `train round N: job ...` every ~30 min, about 4 pieces (one epoch over ~46K rows).
3. `scaling adapters`, `trained; answering`, one answer job.
4. `DONE: answers ready ...` — the main session then judges the answers with three judges and releases a winner.

Expected finish: ~01:00–02:00 CEST Oct 23.

## 4. Every 30 minutes

```bash
S=/leonardo_work/EUHPC_D30_031/shared_split
# run on the login node recorded in $S/ctl/LOGIN_NODE_$USER
pgrep -af "^bash $S/ctl/run_round.sh"          # both rounds alive (until DONE)
squeue --me -o '%.10i %.16j %.8T %.10M %.4D %R'
sacct -S now-1hours -X -n -o jobid,jobname%18,state,elapsed
tail -2 $S/ctl/sh24/controller.log $S/ctl/sh25/controller.log
ls -la --time-style=+%T $S/ctl/sh2?/HEARTBEAT     # updated within the last ~2 min while a job runs
df -h ~ | tail -1                                 # home must stay below 100%
```

Then give the user a 3–5 line status.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| A controller died (not in `pgrep`, no `DONE`/`STOP` line) | Restart it the same way (section 3). It re-attaches to its running job and resumes. |
| `STOP: cannot read the round 20 model` / `Permission denied` under `runs/a08trc09` | Report to the user; the main session fixes permissions. Do not copy or chmod anything there. |
| `Disk quota exceeded` / `Engine core initialization failed` | Check `df -h ~`. Delete only your own `~/.cache/vllm` or `~/.triton`, never anything else. |
| `submission refused (QoS slots busy)` | Normal while both slots are busy; it retries. |
| A job `FAILED` within 1 minute | Read `$S/logs/<name>-<jobid>-0.err` and the vLLM log in `$S/code/slurm/logs/`; report. The controller retries; 3 failures in a row stop it. |
| `STOP: training raised an exception` | Report the traceback from `$S/ctl/<round>/controller.log` to the user; do not retry blindly. |
| Login node refuses SSH | Use another login node; your processes still run on the old one (check with `ls -la $S/ctl/sh2?/HEARTBEAT`). |

## 6. After rounds 24/25

When both logs show `DONE`, tell the user. The main session judges them and decides on a release. **New tasks will be added below this line by the main session; re-read this file (`git pull`) after each `DONE`.**

## 7. Task queue (updated by the main session)

- [ ] `sh24`, `sh25` — section 3 (started: ______)
