# Running DistillKit on Leonardo (CINECA)

Everything runs **on the cluster, offline**: weights are downloaded once on a login node, and each job
serves its own models with vLLM on its own GPU (on `127.0.0.1`, on a port derived from the job id, so
teams sharing the node don't collide). No external API is used.

| Role | Model | GPUs | License |
|---|---|---|---|
| Teacher | `Qwen/Qwen3-32B-AWQ` (vLLM) | 1 | Apache 2.0 |
| Student | `Qwen/Qwen3-4B-Instruct-2507`, QDoRA | 1 per seed (2 with FSDP) | Apache 2.0 |
| Judge | `openai/gpt-oss-20b` (vLLM) | 1 (shared with the student's answers) | Apache 2.0 |
| Cross-check judge | `google/gemma-4-26B-A4B-it` (vLLM) | 1 | Apache 2.0 |

## 1. One-time setup (login node)

```bash
git clone <repo-url> $WORK/alembic && cd $WORK/alembic
$EDITOR slurm/env.sh          # set SBATCH_ACCOUNT (saldo -b), and SBATCH_RESERVATION if we get one
bash slurm/setup_login.sh     # uv, venvs, vLLM, ~95 GB of weights into $WORK/hf_cache, llama.cpp, tests
```

## 2. Day-1 smoke tests (debug QOS, 30 min each)

```bash
sbatch slurm/smoke_teacher.sbatch            # teacher loads; top-20 logprobs + token ids; tool calls; tokens/s
sbatch slurm/calibrate.sbatch                # can we trust gpt-oss-20b? accuracy on known-label pairs
JUDGE=cross sbatch slurm/calibrate.sbatch    # same for Gemma 4
```

Read `slurm/logs/dk-smoke-teacher-<id>.out` (prints `SMOKE TEST: PASS`), and
`runs/qwen3_4b_qdora/calibration_summary_{gpt-oss,gemma4}.json`. Use the tokens/s figure to size the
10k generation and `teacher.concurrency`.

## 3. Pipeline

```bash
bash slurm/submit.sh            # generate -> train (one job per seed) -> evaluate (per seed) -> export
bash slurm/submit.sh train      # data already generated
bash slurm/submit.sh evaluate   # adapters already trained
JUDGE=cross sbatch slurm/evaluate.sbatch      # cross-check judge on the same cached answers
python -m distillkit.aggregate runs/qwen3_4b_qdora              # mean ± std over seeds
python -m distillkit.aggregate runs/qwen3_4b_qdora --tag gemma4
```

| Job | Time | What it does |
|---|---|---|
| `generate.sbatch` | ≤ 24 h, 1 GPU | teacher on vLLM → `generate` (resumable: resubmit if killed) → `verify` |
| `train.sbatch` | ≤ 8 h, array, 1 GPU each | QDoRA per seed in `runs/qwen3_4b_qdora/seed_<n>/` (shares `verified.jsonl`) |
| `train_fsdp.sbatch` | ≤ 8 h, 2 GPUs | one run with FSDP (untested on Leonardo yet: try it on the debug QOS first) |
| `evaluate.sbatch` | ≤ 4 h, array, 1 GPU each | `answer` (base + student, cached) → judge on vLLM → `evaluate` |
| `export.sbatch` | ≤ 2 h | merge → GGUF → imatrix → quants for one seed; `ollama create` runs on the laptop |

Seeds come from `SEEDS` in `env.sh`; the `--array` ranges in `train.sbatch`/`evaluate.sbatch` must
match (a test checks this, and `submit.sh` sets `--array` from `SEEDS`).

## Rules on a shared node

- Ask for 1 GPU per job unless a job needs more; arrays spread seeds over free GPUs.
- Generation appends results as it goes; a job killed at its wall time loses nothing.
- Heavy jobs off-peak where possible; check usage with `squeue --me` and storage with `cindata`.

## Still to confirm with CINECA / the mentors

Account and any hackathon reservation; whether compute nodes have internet (the scripts assume not);
whether Booster jobs may run without a GPU (export only needs CPUs); `$WORK` quota (~95 GB of weights
plus runs).
