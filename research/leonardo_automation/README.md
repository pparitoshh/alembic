# Leonardo automation (Oct 17–18, 2026)

These controllers run unattended on the Leonardo login node. They chain data generation, training, evaluation and comparison through the debug QoS, which allows 2 jobs and 32 GPUs per user. On the server they live in `$R/ctl/<name>/`, where `R=/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017`. They run as `setsid nohup` processes, not under tmux, because a tmux server keeps its first session's command line, so a `pkill -f` aimed at one controller once killed the server and every controller with it.

| Controller | What it does |
|---|---|
| `rag65v3` | Round 3: teacher regenerates the short B7 answers, then RAFT training (3 seeds), adapter scaling (×1.0/×0.5/×0.2), answer, judge and aggregate. |
| `factory_v1` | All-factory batch 1 (`ai_factory_all_v1`, 200 shards): generate/review jobs, then `merge_factory_v1.py`, then `next.sh` → round 4. |
| `factory_v2` | Top-up batch (`ai_factory_all_v2`, 100 shards). It uses only the second debug slot while batch 1 is unmerged and yields to pending round jobs. It merges with batch 1 as an existing bank. |
| `rag65v4` | Round 4: the better round-2/3 set plus the batch-1 release. It trains, evaluates, then re-judges earlier rounds in the same run for paired bootstrap comparisons (`paired.py`). `next.sh` → round 5. |
| `rag65v5` | Round 5: the round-4 set plus the top-up release. Same evaluation, paired against rounds 3 and 4. |
| `watchdog` | Restarts any unfinished controller whose process died, unless it logged ` STOP: `. |

`dryrun.sh` exercises the merge and the round-4/5 data builders on the current shards, writing to a separate release directory.

Every stage resumes from files. Factory controllers keep in-flight job ids in `inflight/` and forget a job only when sacct reports a final state. Round controllers adopt a running job instead of submitting a duplicate.

## Checks and experiments (`checks/`, Oct 20–24)

Single-file controllers that live directly in `$R/ctl/` on the server. Each one judges, confirms or prepares one result
and logs to `ctl/<name>.log`.

| Group | Scripts | What they do |
|---|---|---|
| Release gates | `vs_release.sh`, `auto_release.sh`, `vs_newret.sh` | A round against the current release (`ctl/RELEASE`) in one 3-judge run plus majority, with the default retriever (`vs_release`) or both on the improved retriever (`vs_newret`) |
| Confirmations | `confirm_op235.sh`, `confirm_soup27.sh`, `lb_confirm.sh`, `kd_gemma.sh`, `r11_gemma.sh` | The same answers judged again in a fresh run, against judge noise and the winner's curse |
| Leaderboard | `lb_add_r20.sh`, `lb_add_r27.sh`, `lb_b768.sh`, `lb_newsearch.sh`, `lb_newsearch_full.sh` | Adding releases, the equal 768-token budget (Leaderboard A) and the improved-retriever run (Leaderboard B) |
| Soups | `soup32.sh`, `soup32old.sh`, `soup38.sh`, `soup42.sh`, `soup43.sh` | Uniform weight soups of seed adapters merged into round 27 or 20, answered for the gates |
| Laptop files | `gguf_soup27_eval.sh`, `gguf_q56_eval.sh`, `gguf_s17_eval.sh`, `release43.sh` | GGUF quantization checks (Q4_K_M to Q8_0, 1.7B vs 4B) and release preparation |
| GOAL.md studies | `vs_ablation.sh`, `rs37_score.sh`, `ragc.sh` | Filtering and curriculum ablations, teacher scoring for round 37, retriever variant c |
