# DistillKit

**Modular teacher–student distillation for local, domain-specific assistants.**

DistillKit turns a large open-weight **teacher** model into a compact **student** that runs on a laptop. The student learns from teacher-generated question/answer pairs (sequence-level distillation). No logits are needed and the teacher and student don't have to share a tokenizer, so any teacher that emits text works with any student.

The demo use case is an **HPC Assistant**: a ≤2B-parameter student specialised in Slurm, CUDA, MPI and profiling. It runs offline on a laptop CPU in under 4 GB of RAM.

> **Status:** working prototype, built for the [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475) (Oct 6–29, 2026). The four stages `generate → verify → train → evaluate` run end to end on a laptop (RTX 2060 6 GB). Export (GGUF/Ollama) is not built yet.

## Results so far (toy run)

Qwen3-0.6B + LoRA, trained on 120 teacher answers about Slurm, judged against the untrained base model on held-out questions by a different model family (both answer orders):

| Iteration | Change | Judge win rate vs. base |
|---|---|---|
| 0 | first run (lr 2e-4, rank 64) | 27.5% (loops, never stops) |
| 1 | gentler LoRA (lr 5e-5, rank 16) | 45.0% |
| 2 | 3x data, short-answer prompt | 42.5% |
| 3 | lr 1e-4, 3 epochs | 77.5% (10 questions, 1 seed) |
| 4 | same setting, 3 seeds, 26 questions | **64.1% ± 4.8** (range 59.6–69.2%) |

The goal (> 60%) is met on average but only just; see the [full report](docs/report_iterations_0-4.md) for what moved the win rate, seed-level results and caveats.

## Quick start

```bash
uv sync --extra train              # plain `uv sync` removes torch/trl/peft
echo 'OPENCODE_API_KEY=...' > .env # teacher + judge endpoint (any OpenAI-compatible API works)
uv run distillkit generate -c configs/iter3.yaml   # teacher writes questions and answers
uv run distillkit verify   -c configs/iter3.yaml   # dedup, flag and bash-syntax checks
uv run distillkit train    -c configs/iter3.yaml   # LoRA SFT of the student
uv run distillkit evaluate -c configs/iter3.yaml   # student vs. base, judged remotely
uv run pytest -q
```

Outputs go to the config's `run_dir` (e.g. `runs/iter3/`, git-ignored). Each experiment has its own config: `configs/toy.yaml` (iteration 0) and `iter1.yaml` to `iter4_s*.yaml`. API calls retry transient errors with exponential backoff.

## Pipeline

```
 seed docs ──► teacher inference ──► verification ──► student training ──► quantize/export ──► evaluate
 (Slurm,       (vLLM, multi-GPU,      (dedup, sbatch/    (TRL SFT, FSDP/      (GGUF Q8_0 /        (held-out docs,
  CUDA, MPI)    N answers/question)    nvcc checks,       DeepSpeed,           Q4_K_M + imatrix,   exec checks,
                                       LLM judge)         optional RAFT/DPO)   Ollama)             LLM judge)
```

| Module | Responsibility |
|---|---|
| **Teacher** | Load any Hugging Face model via vLLM, or call an OpenAI-compatible endpoint |
| **Data generation** | Document-grounded, diversified Q&A generation (persona × task type × difficulty) |
| **Verification** | Deduplication, executable checks, LLM-as-judge, eval decontamination |
| **Student training** | Full fine-tuning or LoRA with TRL; multi-node FSDP/DeepSpeed |
| **Export & eval** | GGUF quantization, Ollama packaging, accuracy/size/speed evaluation |

## Documentation

- **[GOAL.md](GOAL.md):** mission, goals, success metrics, deliverables, team roles and timeline.
- **[RESEARCH.md](RESEARCH.md):** state-of-the-art survey and design rationale (data generation, filtering, the RAG/RAFT knowledge problem, training, quantization, evaluation).
- **[docs/training.md](docs/training.md):** plain-language guide to how training works: one sample, what the student predicts, micro-batches and optimizer steps, and which weights are frozen or trained.
- **[docs/report_iterations_0-4.md](docs/report_iterations_0-4.md):** comparison of all iterations, seed-level testing, and which parameters moved the win rate.
- **[docs/iter_0_learning.md](docs/iter_0_learning.md)** to **[iter_3_learning.md](docs/iter_3_learning.md):** per-iteration write-ups: what changed, results, lessons.
- **[PROGRESS.md](PROGRESS.md):** running log against [PLAN.md](PLAN.md).

## Planned stack

Python · PyTorch · Hugging Face Transformers / TRL · vLLM · FSDP / DeepSpeed · llama.cpp (GGUF) · Ollama

## Contributing

Work happens on the `dev` branch; `main` holds reviewed milestones. Open a pull request against `dev`.

## License

[MIT](LICENSE) © 2026 DistillKit contributors
