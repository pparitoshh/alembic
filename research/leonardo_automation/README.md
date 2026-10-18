# Leonardo automation (Oct 17–18, 2026)

These controllers run unattended on the Leonardo login node. They chain data generation, training, evaluation and comparison through the debug QoS, which allows 2 jobs and 32 GPUs per user. On the server they live in `$R/ctl/<name>/`, where `R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017`. They run as `setsid nohup` processes, not under tmux, because a tmux server keeps its first session's command line, so a `pkill -f` aimed at one controller once killed the server and every controller with it.

| Controller | What it does |
|---|---|
| `rag65v3` | Round 3: teacher regenerates the short B7 answers, then RAFT training (3 seeds), adapter scaling (×1.0/×0.5/×0.2), answer, judge and aggregate. |
| `factory_v1` | All-factory batch 1 (`ai_factory_all_v1`, 200 shards): generate/review jobs, then `merge_factory_v1.py`, then `next.sh` → round 4. |
| `factory_v2` | Top-up batch (`ai_factory_all_v2`, 100 shards). It uses only the second debug slot while batch 1 is unmerged and yields to pending round jobs. It merges with batch 1 as an existing bank. |
| `factory_v3` | Coverage batch (`ai_factory_all_v3`, 130 shards): only chunks batches 1-2 never used. It starts once the top-up has nothing in flight. |
| `factory_v4` | Final batch (`ai_factory_all_v4`): a third angle set over every relevant chunk, prose and ask-back only. It starts once v3 has nothing in flight. |
| `rag65v4` | Round 4: the better round-2/3 set plus the batch-1 release (42,106 rows). It trains, builds the strengths and seed soups, evaluates, then re-judges earlier rounds in the same run for paired bootstrap comparisons (`paired.py`). |
| `rag65v5` | Round 5: the round-4 set plus the top-up release (~58.6K rows). It runs standalone from the top-up merge, and `next.sh` → round 6. |
| `rag65v6` | Round 6 (final): the round-5 set plus the coverage and final releases (100K+ rows). It waits for round 5's scores and both merges. |
| `watchdog` | Restarts any unfinished controller whose process died, unless it logged ` STOP: `. Controllers are matched by name. |
| `priority_submit.sh` | Reserves the next free debug slot for a one-off job (`ctl/PRIORITY`). |

`dryrun.sh` exercises the merge and the round-4/5 data builders on the current shards, writing to a separate release directory.

Every stage resumes from files. Factory controllers keep in-flight job ids in `inflight/` and forget a job only when sacct reports a final state. Round controllers adopt a running job instead of submitting a duplicate.

## Guards learned the hard way (Oct 18)

- Factory jobs use at most 5 nodes, so a 3-node training job always fits under the 8-node limit, including between training pieces.
- Job ids must be numeric. Error text from squeue or sbatch is never waited on, and running jobs are adopted by name instead of resubmitted.
- Answers are checked per arm file before judging. Adapters must have weights. MERGED is logged only for a freshly written release.
- Jobs are not cancelled by hand. A job that lands on nodes released by a just-cancelled job can be killed during node cleanup; three jobs were lost that way.
- In worker scripts under `set -e`, no `[ test ] && cmd` as the last command inside `$(...)`, and no bare `wait` once a server runs in the background.
