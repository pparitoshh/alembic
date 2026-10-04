# Research results

Measured results that back the claims in [GOAL.md](../GOAL.md), committed so they survive (`runs/` is
git-ignored). Design rationale and literature are in [RESEARCH.md](../RESEARCH.md); the running log is
[PROGRESS.md](../PROGRESS.md).

| Folder | Contents | Produced by |
|---|---|---|
| [`laptop/`](laptop/) | The benchmark machine: CPU, RAM, GPU, OS, software versions | `python -m distillkit.machine --out research/laptop/<name>` |
| [`bench/`](bench/) | Laptop speed and peak RAM per GGUF quant level (G3, G6) | `distillkit bench` (results in `run_dir/export/bench.{json,md}`, copied here) |
| [`judge_calibration/`](judge_calibration/) | How far each judge can be trusted, on known-label pairs | `distillkit calibrate` (`run_dir/calibration_summary_<tag>.json`, copied here) |

Each result folder is named `<date>_<what>` and has a README with the setup and how to read it.
Machine-dependent numbers (speed, RAM) are only comparable on the same machine and settings: every
bench report embeds its machine record.

Not recorded: RAM type and speed (need `sudo dmidecode -t memory`), which matter for CPU generation
speed since it is memory-bandwidth bound.
