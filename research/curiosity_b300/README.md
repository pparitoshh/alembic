# Curiosity (8 × NVIDIA B300): automated distillation loop (Oct 26)

The team's Curiosity allocation is one DGX B300 node (8 GPUs, team QoS). These scripts run the 4B student's
train → evaluate → release loop there without manual steps. The student stays Qwen3-4B.

| Script | Role |
|---|---|
| `agent.sbatch`, `watchdog.sh` | One long job holds the 8 GPUs and runs `~/cp/queue/*.sh` in order; the login-node watchdog resubmits it |
| `lib.sh` | Shared task environment: vLLM servers (`serve`), cleanup of every server between tasks, CUDA 13 toolchain for FlashInfer/DeepGEMM |
| `dl.sh` | Model downloads (round 27, Qwen3-4B, Qwen3-235B FP8, gpt-oss-20b/120b, Gemma 4) |
| `build_prompts.py` | Training prompts from every verified bank: held-out families and near-test questions removed, app excerpts (retriever e) in the user turn |
| `teacher_gen.py` | Qwen3-235B (FP8) answers with top-20 log-probabilities, 4 replicas × TP2 |
| `train_kd.py` | Full-parameter distillation of the 4B student (forward KL on the teacher's top-20 + CE on its tokens, 8-GPU data parallel) |
| `exp_kd.sh` | One experiment: prompts → teacher → training → `eval.sh` |
| `eval.sh`, `answer.py`, `judge_ref.sh`, `judge_smoke.sh` | The leaderboard evaluation (equal-1536, retriever e, reviewed sets, 3 judges + majority, paired vs round 27) |
| `autopilot.py`, `autopilot.sh` | Picks the next experiment from the results; a win (mean ≥ +0.02 over the best, v3 not below −0.015) requests a release |
| `release_build.sh`, `hf_push.py`, `release_watch.sh` | Release: GGUF build on Curiosity, push from the laptop (token over ssh stdin, never stored on the cluster). Each win is published as the next version (1.N), the previous `main` is kept on branch `v1.N`, and the card never shows internal run names |
| `kd1_leonardo_eval.sh` | Confirms a Curiosity result on Leonardo's own leaderboard pipeline |
| `kdall.sh` | Distils the best student on the union of every teacher set so far (no new teacher generation) |
| `kdgpu.sh` | A fresh round with prompts weighted to GPU, MPI and the software environment (`PROMPT_MIX` in `build_prompts.py`) |
| `slurm_exec_answers.sh` | The newest models answer the real-Slurm execution tasks (`research/slurm_exec_eval`) |

## Results (Oct 26)

- **kd1** (round 27 + full-parameter distillation on 27,367 Qwen3-235B answers, α 0.8, lr 5e-6, 1 epoch):
  Curiosity evaluation vs round 27: v2 +0.011, v3 +0.021, d1 +0.040 (mean +0.024). Leonardo confirmation: v2 −0.015,
  v3 −0.004, d1 +0.012 (mean −0.002). Level overall, GPU/MPI up in both; 20/30 on the real-Slurm tasks
  (`research/slurm_exec_eval`) vs 17/30 for round 27. Released as Hugging Face `main` (round 27 kept on `round27`).
- Noise check: round 27 re-answered on Curiosity scored +0.014 mean against its own Leonardo answers.

## Results (Oct 27)

Public names: ClusterPilot-4B 1.0 = round 27, 1.1 = `kd1`, 1.2 = `kd2`, 1.3 = `kdall`.

| Run | What | B300 mean vs round 27 (same run) | Reviewed leaderboard (Leonardo): mean / Slurm / Slurm dev / GPU-MPI | Real Slurm |
|---|---|---|---|---|
| `kd1` (1.1) | 27,367 teacher answers on round 27 | 0.8175 vs 0.7935 | 0.815 / 0.885 / 0.830 / 0.729 | 20/30 |
| `kd2` (1.2) | 54,568 fresh answers on `kd1` | 0.838 vs 0.793 | 0.837 / 0.921 / 0.864 / 0.726 | 20/30 |
| `kd3` | 54,517 fresh answers on `kd2` | 0.849: +0.011 over `kd2`, below the +0.02 release bar | | 21/30 |
| `kdall` (1.3) | union of the `kd1`–`kd3` sets (136,452 answers) on `kd2` | **0.867** vs 0.793 | **0.847 / 0.933 / 0.856 / 0.752** | **21/30** |

- On the reviewed Slurm set, 1.3 (0.933) is above Qwen3-Next-80B (0.911) and gpt-oss-120b (0.926), behind Qwen3-235B (0.973)
  and Gemma-4-26B (0.935). Without documentation search, 1.2 scored 0.725 mean (Slurm 0.810).
- 1.2 was also confirmed on Leonardo's original pipeline (AWQ judge): mean 0.844 vs 0.801 for round 27.
- Training on the union of teacher sets beat another fresh round of the same size: more teacher data per step, no new
  teacher time.

## Lessons (B300)

- Pin every CUDA 13 compiler component in the venv (`nvidia-cuda-nvcc`, `nvidia-nvvm`, `nvidia-cuda-cccl`, `-crt`) to the
  runtime's version (13.0), add plain `lib*.so` links, and point `CUDA_HOME` at the venv's `nvidia/cu13`; mixed versions
  break FlashInfer/DeepGEMM JIT ("CUDA compiler and toolkit headers are incompatible", "Unsupported .version 9.4").
- `GROUPS` is a bash builtin (the user's group ids); using it as a variable started 959 teacher processes.
- Servers started with `serve ... &` are not in the parent's PID list: clean up by pattern between tasks.
