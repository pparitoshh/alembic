# Judge calibration: OpenCode judges (2026-10-04)

Known-label pairs: `data/eval/judge_calibration.jsonl` (24 constructed pairs, 10 A / 10 B / 4 ties; one answer has a single deliberate defect). Each pair judged in both answer orders with the eval's judge prompt (`distillkit calibrate`).

| Judge | Accuracy | Position consistency | Unparsed | invented_flag | length_bait | tie | wrong_command | wrong_fact | wrong_math | wrong_semantics | wrong_syntax |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash | 1.000 | 1.000 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| minimax-m3 | 0.979 | 0.958 | 0 | 1.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| kimi-k3 | 0.958 | 0.917 | 0 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| deepseek-v4.1-flash | 0.938 | 0.958 | 0 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Reading:** every judge caught every single-defect error; only the ties separate them. The set is a floor, not a ranking: ranking strong judges needs human-labelled real student-vs-base pairs. The judges chosen for Leonardo (gpt-oss-20b, Gemma 4 26B-A4B) are not on OpenCode; they are calibrated there with `slurm/calibrate.sbatch`.
