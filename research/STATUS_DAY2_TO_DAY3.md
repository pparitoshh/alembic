# Status: day 2 to day 3 (Oct 27 morning to Oct 28 morning)

Team Alembic, ClusterPilot-4B. Scores are win rates against closed-book Qwen3-4B (0.50 = the base model with no help),
judged by three AI judges (gpt-oss-20b, Gemma 4, Qwen3-235B) with a majority vote. Test sets: v3 = Slurm (146 reviewed
questions), v2 = Slurm dev, d1 = "GPU/MPI" (106 questions). "B300 test" = the evaluation on Curiosity (8 x B300) against
version 1.0 in the same run; "leaderboard" = the reviewed leaderboard on Leonardo, same setup for every model.

## Summary

- **Best model: ClusterPilot-4B 1.3** (Hugging Face `main`). Leaderboard: Slurm **0.933** (3rd of the 7 open models we
  tested, level with Gemma-4 26B at 0.935, above gpt-oss-120b 0.926 and our 80B teacher 0.911), mean 0.847, GPU/MPI
  0.752. Without documentation search: Slurm 0.858. Real Slurm tasks: 21/30.
- **14 more training runs** after 1.3 (new data, learning rates, MPI/NCCL rounds, model soups): none beat it. The best
  came within 0.004 (0.863 vs 0.867 on the B300 test).
- **Data:** 5.5 lakh to **7.18 lakh** verified Q&A, including **9,938** new MPI/NCCL rows from two focused batches.
- **Measured:** NCCL and MPI between two Leonardo nodes (NCCL all-reduce 84 GB/s, 19x slower without InfiniBand).
- **Blocked:** the Leonardo project budget ran out (152,285 of 144,000 h, 105.8%), so GPU jobs are refused; all work
  is on the B300s now.
- **Now:** a targeted round on Slurm requeue/signals/checkpoint/arrays, the d1 groups where we trail most.

## Model versions on Hugging Face

| Version | What | Leaderboard mean | Slurm | GPU/MPI | Real Slurm |
|---|---|---|---|---|---|
| 1.0 | soup of 12 distillation runs (round 27) | 0.810 | 0.907 | 0.698 | 17/30 |
| 1.1 | 1.0 + 27k Qwen3-235B teacher answers on B300 | 0.815 | 0.885 | 0.729 | 20/30 |
| 1.2 | 1.1 + 55k fresh teacher answers | 0.837 | 0.921 | 0.726 | 20/30 |
| **1.3** | 1.2 trained on the union of all three teacher sets (136k answers) | **0.847** | **0.933** | **0.752** | **21/30** |

## Training runs (B300 test, gain over 1.0 in the same run)

| Run | What we tried | Slurm dev | Slurm | GPU/MPI | Mean | Result |
|---|---|---|---|---|---|---|
| kd2 (1.2) | fresh 55k teacher answers on 1.1 | +0.045 | +0.029 | +0.059 | 0.838 | passed, released |
| kd3 | another fresh 55k on 1.2 | +0.110 | +0.027 | +0.028 | 0.849 | failed (+0.011, under the +0.02 bar) |
| **kdall (1.3)** | 1.2 trained on the union of the kd1-kd3 teacher sets | +0.095 | +0.039 | +0.085 | **0.867** | **passed, released** |
| kdgpu | 65% GPU/MPI/env prompts | +0.019 | +0.010 | +0.026 | 0.812 | failed: GPU-heavy data hurt everything |
| kdall2 | union + the GPU-heavy set | +0.053 | +0.051 | +0.061 | 0.849 | failed |
| kd4 | fresh 60k on 1.3 | +0.080 | +0.036 | +0.073 | 0.856 | failed |
| kdall3 | union of kd1-kd4 | +0.091 | +0.043 | +0.038 | 0.851 | failed |
| kd1_lr1e-5 | 1.3 + kd1 answers, higher learning rate | +0.095 | +0.043 | +0.068 | 0.862 | failed |
| kd1_a1 | pure distillation (alpha 1.0) | +0.061 | +0.039 | +0.080 | 0.854 | failed |
| kd1_lr2e-6 | lower learning rate | +0.061 | +0.021 | +0.040 | 0.834 | failed |
| kd1_big | 117k fresh teacher answers on 1.3 | +0.068 | +0.044 | +0.073 | 0.855 | failed |
| kdall4 | 1.3 recipe + v300 MPI/NCCL answers | +0.072 | +0.062 | +0.075 | 0.863 | failed (best Slurm gain so far) |
| kdall5 | 1.3 recipe + v300 and v400 MPI/NCCL answers | +0.061 | +0.050 | +0.028 | 0.840 | failed |
| kd1_base2ep | base Qwen3-4B, 2 epochs | -0.008 | +0.017 | +0.028 | 0.806 | failed |
| soup2 | weight average of kdall and kdall4 | +0.053 | +0.027 | +0.043 | 0.835 | failed |
| soup4 | average of kdall, kdall4, kd4, kd1_big | +0.064 | +0.034 | +0.075 | 0.852 | failed |

A new model is published only when its B300 mean beats the best by +0.02 with Slurm not lower; the B300 test varies by
about 0.015 between runs of the same model.

## What worked and what did not

Worked:

- **Training on the union of teacher sets** (1.3): more teacher answers per step from Qwen3-235B, no new teacher time.
- **Releases with version names** (1.0 to 1.3, older versions kept as branches) and Ollama `template`/`system`/`params`
  files at the repo root, so `ollama pull hf.co/...` uses our template (found by Paritosh: tool calls had gone into the
  thinking field).

Did not work:

- **More of the same data** after the 136k union: kd4, kd1_big and kdall3 all scored lower.
- **GPU-heavy prompts:** the GPU/MPI/env bucket is mostly modules, pip and conda questions; it hurt every test set.
- **MPI/NCCL data for the GPU/MPI test:** see the next section.
- **Learning rate, pure distillation, 2 epochs from the base model, model soups:** all lower.

## Why the GPU/MPI score did not move

d1 (106 questions) by group, 1.3 against Qwen3-235B on the leaderboard:

| Group | Questions | 1.3 | Qwen3-235B |
|---|---|---|---|
| Slurm requeue and signals | 36 | 0.81 / 0.76 | 0.95 / 0.88 |
| Held-out cluster (Aalto Triton: GPU jobs, MPI launch, profiling, modules, conda) | 40 | 0.61 | 0.73 |
| Slurm job arrays | 30 | 0.90 / 0.90 | 0.95 / 0.93 |

d1 has almost no general MPI or NCCL questions. Its MPI/GPU questions are about a cluster whose documentation we keep
out of training on purpose, so general MPI/NCCL knowledge cannot raise it. The clear, trainable gap is Slurm requeue,
signals and checkpointing.

## Data

| Item | Day 2 | Day 3 |
|---|---|---|
| Verified Q&A rows | about 5.5 lakh | **717,903** |
| MPI/NCCL batches | none | v300: 5,508 rows; v400: 4,430 rows |
| New source pages (gap-6 crawl) | none | 310 (Intel MPI, MVAPICH/OSU, HPC-X, libfabric, mpitutorial) |

## Measurements on Leonardo (two Booster nodes, `research/commbench/`)

| Test | Result |
|---|---|
| NCCL all-reduce, 8 GPUs on 2 nodes, default | 84 GB/s |
| same, `NCCL_ALGO=Ring` | 49 GB/s |
| same, GPUDirect RDMA off | 39 GB/s |
| same, InfiniBand off | 4.5 GB/s (19x slower) |
| MPI GPU to GPU between nodes / inside a node (NVLink) | 12.3 / 87 GB/s |
| MPI latency between nodes, CPU / GPU memory | 1.4 / 2.0 us |

## Operations

- Hackathon reservation: from Oct 27, all our GPU jobs ran in `s_tra_ncc1` (organisers' request); the flag comes from
  `shared_split/ctl/resv_flag.sh` and the factory job file.
- Leonardo budget: exhausted at 03:51 CET Oct 28; our factory is paused (`shared_split/ctl/PAUSED_a08trc09.txt`).
- Curiosity: the agent job holds the team's 8 B300 GPUs; the team group `eai-team04` can read `~/cp/logs`, `runs`,
  `autopilot` and `queue` of the agent account.

## Running now (Oct 28)

1. `kdreq`: Qwen3-235B answers for the **11,625** bank questions about Slurm requeue, signals, preemption,
   checkpoint/restart, `--time-min`, `--open-mode`, job arrays and dependencies (near-test questions excluded).
2. `kdall6`: the 1.3 recipe plus that set. Verdict around 14:00 CET; a win is published as ClusterPilot-4B 1.4.

## Next

1. If `kdall6` lifts the requeue part: raise that set's weight, or train it from 1.3.
2. A practical MPI/GPU round: Slurm job scripts for MPI and GPU (srun/mpirun launch, GPU binding, CUDA modules,
   profiling commands), not general MPI/NCCL theory.
3. When Leonardo has hours again: leaderboard check and real-Slurm test of any new winner.

Details: `research/curiosity_b300/HANDOVER_2026-10-28.md` (PR #80).
