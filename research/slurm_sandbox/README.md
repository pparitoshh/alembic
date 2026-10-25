# Slurm sandbox model evaluation

Goal: measure how well the distilled student
(`hf.co/VaibhavKKM/hpc-assistant-qwen3-4b`) answers real Slurm tasks, and
compare it against the old student (`hpc-assistant-x02`). Every answer is
executed unchanged in a real Slurm 23.11 container, so the verdicts come from
Slurm itself, not from a human reading text.

## Method

1. Write the prompts and the expected answers first. The expected answers use
   plain Slurm knowledge with the main required flags.
2. Ask the model with one fresh call per prompt:
   `ollama run <model> "<prompt>" --think=false`. One run per prompt. Save the
   raw output.
3. Run the model answer in the Docker Slurm container as user `ubuntu`
   (`docker exec -u ubuntu -e USER=ubuntu`). Only file names are adapted. No
   model error is fixed. Record the real output or error.
4. Rate each answer: correct, partly correct, or wrong, with one reason.

## Setup

The container recipe lives in `docker/` (configs, shim, start script,
`docker/setup.md`). Highlights: fake GPUs on `/dev/fakegpu0-3`, an
LD_PRELOAD shim for the cgroup detection of Slurm 23.11, and slurmdbd +
MariaDB for accounting. Without accounting, Slurm 23.11 pends every job with
`InvalidAccount`.

## Limits

- Fake GPUs (char devices on major 195), one node, no real CUDA.
- Sampling varies: each prompt runs once, one run per prompt.
- Round 2 used 10 prompts; round 1 was a 4-prompt smoke test.
- The container has no `module` command and no mail binary, so those two
  answers can only be tested up to submission.

## How to rerun

1. Build the container with `docker/setup.md`.
2. Feed each line of `prompts.txt` to the model, one call each; keep the raw
   output next to the date folder in `results/`.
3. Execute each answer in the container as user `ubuntu` and capture the
   output.
4. Compare with `expected.md`, rate, and write `results.json` and
   `results.md` in a new `results/<date>/` folder.

## Scores

| Round | Date | Prompts | Correct | Partly | Wrong |
|---|---|---|---|---|---|
| 1 (smoke, x02 vs new) | 2026-10-25 | 4 | new 3, old 1 | 0 | new 1, old 3 |
| 2 (10 prompts, new model) | 2026-10-25 | 10 | 6 | 2 | 2 |

Round 2 details are in `results/2026-10-25/results.md`. The repeated weakness
is the cancel-by-name command (`scancel -n train`). The sbatch header,
dependencies, mail flags, srun, sinfo, and scontrol show job forms are solid.
