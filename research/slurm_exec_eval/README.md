# Execution-based Slurm evaluation on a real cluster (Oct 26)

Extends Paritosh's Docker sandbox evaluation (branch `research/slurm-sandbox-eval`: 10 prompts, answers run in a
Slurm 23.11 container) to 30 tasks graded against Leonardo's real Slurm.

- `tasks.jsonl`: his 10 prompts plus 20 new ones (arrays, requeue, signals, dependencies, GPU/MPI directives,
  `scontrol update`, `sacct`/`squeue` filters, OpenMP), each with required and forbidden patterns.
- `answer_tasks.py`, `answer_task.sh`: every model answers with the evaluation system prompt, no documentation excerpts
  (as in the sandbox eval), greedy (run on Curiosity, 8 × B300).
- `grade.py`: two checks per answer, on a Slurm login node.
  - **Rules:** every required pattern is in the answer's code; no forbidden pattern (e.g. `--array=1-10:3`) in the fenced code
    the user would run.
  - **Slurm:** the answer runs as-is. Scripts and `#SBATCH` lines go through `sbatch --test-only`, `srun` lines through
    `srun --test-only`, and query and job-control commands run against a held fixture array job (start date a week
    away, so a `release` cannot make it run). Refusals caused by the cluster's own limits or configuration (QoS submit
    and CPU caps, disabled requeue, `seff` not installed, partition time limit) are not counted against the answer.
  - Correct = rules pass and Slurm accepts it; partly correct = rules pass but Slurm rejects it, or at least half of
    the rules pass; wrong = otherwise.
- `reference_answers.jsonl`: textbook answers; they score 30/30 (checks the grader, not the models).

## Results (Oct 26, Leonardo, 30 tasks)

| Model | Correct | Partly | Strict | Lenient | Wilson 95% (strict) | Slurm accepted |
|---|---|---|---|---|---|---|
| Qwen3-235B-A22B (FP8) | 26/30 | 2 | 0.87 | 0.90 | [0.70, 0.95] | 27/30 |
| **ClusterPilot-4B kd1 (HF main)** | **20/30** | 6 | **0.67** | **0.77** | [0.49, 0.81] | 22/30 |
| ClusterPilot-4B round 27 | 17/30 | 10 | 0.57 | 0.73 | [0.39, 0.73] | 19/30 |
| gpt-oss-20b | 14/30 | 1 | 0.47 | 0.48 | [0.30, 0.64] | 18/30 |
| Qwen3-4B-Instruct-2507 (base) | 10/30 | 10 | 0.33 | 0.50 | [0.19, 0.51] | 16/30 |

gpt-oss-20b reasons before answering, and 2 of its answers were empty at the 1,024-token limit. Per-task ratings and
Slurm's output for every answer are in `results/2026-10-26/`. Still wrong for most models: `scontrol requeuehold`
(task 14) and the array throttle `ArrayTaskThrottle` (task 30).
