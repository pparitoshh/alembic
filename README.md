# DistillKit

**Modular teacher–student distillation for local, domain-specific assistants.**

DistillKit turns a large open-weight **teacher** model into a compact **student** that runs on a laptop. The student learns from teacher-generated question/answer pairs (sequence-level distillation). No logits are needed and the teacher and student don't have to share a tokenizer, so any teacher that emits text works with any student.

The demo use case is an **HPC Assistant**: a ≤2B-parameter student specialised in Slurm, CUDA, MPI and profiling. It runs offline on a laptop CPU in under 4 GB of RAM.

> **Status:** planning / early prototype, built for the [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475) (Oct 6–29, 2026). No code has landed yet. The design is in the docs below.

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

## Planned stack

Python · PyTorch · Hugging Face Transformers / TRL · vLLM · FSDP / DeepSpeed · llama.cpp (GGUF) · Ollama

## Contributing

Work happens on the `dev` branch; `main` holds reviewed milestones. Open a pull request against `dev`.

## License

[MIT](LICENSE) © 2026 DistillKit contributors
