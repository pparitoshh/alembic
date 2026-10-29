# Curiosity pipeline (B300 cluster, Oct 25-29 2026)

The scripts that trained and evaluated ClusterPilot-4B 1.3 and the later candidates on Curiosity (8 NVIDIA B300,
team QOS eai-team04), copied as they ran from `~/cp` before the cluster access ended. Paths in the scripts are the
Curiosity ones (`/home/eai-jjppv/cp/...`).

| Folder | What it is |
|---|---|
| `bin/` | The task queue (`agent.sbatch`, `watchdog.sh`, `side_submit.sh`), teacher generation (`teacher_gen.py`), distillation (`train_kd.py`: full-parameter KD on 8 GPUs, plain data parallelism with an explicit NCCL all-reduce, bf16 autocast, gradient checkpointing; `kdall.sh`, `exp_kd.sh`), model soups (`soup.py`), the B300 evaluation (`eval.sh`, `judge_*.sh`), QA sweeps, the release build and the Curiosity demo launcher (`cp_demo.sh`) |
| `cb/` | The closed-book round for issue #115 (context distillation of the app's documentation into the model) |
| `mentor/` | Blind key-point scoring of the mentor use cases (`score_mentor.py`) and the with-app answer runner |

The data these scripts produced (Qwen3-235B teacher answers with top-20 log-probabilities, prompts, question banks,
logs and evaluation results; about 60 GB, no model weights) is in the private Hugging Face dataset
`VaibhavKKM/clusterpilot-curiosity-archive`. The released model is `VaibhavKKM/hpc-assistant-qwen3-4b`.
