# DistillKit

**Modular teacher–student distillation for local, domain-specific assistants.**

DistillKit turns a large open-weight **teacher** model into a compact **student** that runs on a laptop. The student learns from teacher-generated question/answer text and tool-calling traces (sequence-level distillation), so any teacher that emits text works with any student. The teacher's top-20 logprobs are saved as well, so logit-level distillation can be added later without re-running the teacher.

The demo use case is an **HPC Assistant**: Qwen3-32B distilled into a **Qwen3-4B-Instruct-2507** student (QDoRA) specialised in Slurm, CUDA, MPI and profiling, with tool calling (job status, submission, logs, GPU availability). Target: runs offline on a laptop CPU in under 4 GB of RAM.

> **Status:** working prototype, built for the [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475) (Oct 6–29, 2026). The four stages `generate → verify → train → evaluate` run end to end on a laptop (RTX 2060 6 GB). Export (GGUF/Ollama) is not built yet.

## Results so far (toy run)

Toy setup, not the final models: Qwen3-0.6B + LoRA, trained on 120 teacher answers about Slurm, judged against the untrained base model on held-out questions by a different model family (both answer orders):

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
uv run distillkit train    -c configs/iter3.yaml   # LoRA / QDoRA SFT of the student
uv run distillkit evaluate -c configs/iter3.yaml   # student vs. base, judged remotely
uv run pytest -q
```

Outputs go to the config's `run_dir` (e.g. `runs/iter3/`, git-ignored). Each experiment has its own config: `configs/toy.yaml` (iteration 0) and `iter1.yaml` to `iter4_s*.yaml`. Configs are validated on load (unknown keys are errors). API calls retry transient errors with exponential backoff. `generate` appends results as they finish, so rerunning it after a crash or a wall-time kill only does the missing work.

**QDoRA and the cluster setup:**

```bash
uv run distillkit train -c configs/toy_qdora.yaml                   # local QDoRA smoke test (Qwen3-0.6B, 4-bit + DoRA)
uv run distillkit all   -c configs/qwen3_4b_qdora.yaml              # Leonardo: Qwen3-32B teacher on vLLM -> Qwen3-4B-Instruct-2507
accelerate launch --config_file configs/accelerate/fsdp.yaml \
    -m distillkit.cli train -c configs/qwen3_4b_qdora.yaml          # multi-GPU QDoRA with FSDP (untested on Leonardo yet)
```

With `teacher.top_logprobs: 20`, the teacher's top-20 logprobs for every answer token go to `run_dir/teacher_logprobs.jsonl.gz`, keyed by row `id`.

## Pipeline

```
 seed docs ──► teacher inference ──► verification ──► student training ──► quantize/export ──► evaluate
 (Slurm,       (vLLM, Qwen3-32B,     (dedup, sbatch/    (TRL SFT, QDoRA,     (GGUF Q8_0 /        (held-out docs,
  CUDA, MPI)    top-20 logprobs)       nvcc, tool-call    FSDP,                Q4_K_M + imatrix,   exec checks,
                                       checks, judge)     optional RAFT/DPO)   Ollama)             LLM judge)
```

| Module | Responsibility |
|---|---|
| **Teacher** | Any model that emits text: Hugging Face model, GGUF, or an OpenAI-compatible endpoint; top-20 logprobs saved |
| **Data generation** | Document-grounded, diversified prose Q&A and tool-calling traces (persona × task type × difficulty) |
| **Verification** | Deduplication, executable checks, tool-call validation, LLM-as-judge, eval decontamination |
| **Student training** | TRL SFT with QDoRA (4-bit base + DoRA adapters), FSDP for multi-GPU |
| **Export & eval** | GGUF quantization, Ollama packaging, accuracy/size/speed evaluation |

## Documentation

- **[GOAL.md](GOAL.md):** mission, final model choices, goals, order of work, success metrics, deliverables, team roles and timeline.
- **[RESEARCH.md](RESEARCH.md):** state-of-the-art survey and design rationale (teacher serving and logprobs, tool-calling data, verification, QDoRA + FSDP, RAG, quantization, evaluation).
- **[docs/training.md](docs/training.md):** plain-language guide to how training works: one sample, what the student predicts, micro-batches and optimizer steps, and which weights are frozen or trained.
- **[docs/report_iterations_0-4.md](docs/report_iterations_0-4.md):** comparison of all iterations, seed-level testing, and which parameters moved the win rate.
- **[docs/iter_0_learning.md](docs/iter_0_learning.md)** to **[iter_3_learning.md](docs/iter_3_learning.md):** per-iteration write-ups: what changed, results, lessons.
- **[PROGRESS.md](PROGRESS.md):** running log of what is done.

## Planned stack

Python · PyTorch · Hugging Face Transformers / TRL / PEFT (DoRA) · bitsandbytes · vLLM · FSDP · llama.cpp (GGUF) · Ollama

## Contributing

Work happens on the `dev` branch; `main` holds reviewed milestones. Open a pull request against `dev`.

## License

[MIT](LICENSE) © 2026 DistillKit contributors
