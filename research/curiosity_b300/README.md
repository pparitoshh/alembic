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
| `release_build.sh`, `hf_push.py`, `release_watch.sh` | Release: GGUF build on Curiosity, push from the laptop (token over ssh stdin, never stored on the cluster) |
| `kd1_leonardo_eval.sh` | Confirms a Curiosity result on Leonardo's own leaderboard pipeline |

## Results (Oct 26)

- **kd1** (round 27 + full-parameter distillation on 27,367 Qwen3-235B answers, α 0.8, lr 5e-6, 1 epoch):
  Curiosity evaluation vs round 27: v2 +0.011, v3 +0.021, d1 +0.040 (mean +0.024). Leonardo confirmation: v2 −0.015,
  v3 −0.004, d1 +0.012 (mean −0.002). Level overall, GPU/MPI up in both; 20/30 on the real-Slurm tasks
  (`research/slurm_exec_eval`) vs 17/30 for round 27. Released as Hugging Face `main` (round 27 kept on `round27`).
- Noise check: round 27 re-answered on Curiosity scored +0.014 mean against its own Leonardo answers.

## Lessons (B300)

- Pin every CUDA 13 compiler component in the venv (`nvidia-cuda-nvcc`, `nvidia-nvvm`, `nvidia-cuda-cccl`, `-crt`) to the
  runtime's version (13.0), add plain `lib*.so` links, and point `CUDA_HOME` at the venv's `nvidia/cu13`; mixed versions
  break FlashInfer/DeepGEMM JIT ("CUDA compiler and toolkit headers are incompatible", "Unsupported .version 9.4").
- `GROUPS` is a bash builtin (the user's group ids); using it as a variable started 959 teacher processes.
- Servers started with `serve ... &` are not in the parent's PID list: clean up by pattern between tasks.
