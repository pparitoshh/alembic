# DistillKit

**Modular teacher–student distillation for local, domain-specific assistants.**

DistillKit turns a large open-weight **teacher** model into a compact **student** that runs on a laptop. The student learns from teacher-generated question/answer text and tool-calling traces (sequence-level distillation), so any teacher that emits text works with any student. The teacher's top-20 logprobs are saved as well, so logit-level distillation can be added later without re-running the teacher.

The demo use case is an **HPC Assistant**: Qwen3-32B distilled into a **Qwen3-4B-Instruct-2507** student (QDoRA) specialised in Slurm, CUDA, MPI and profiling, with tool calling (job status, submission, logs, GPU availability). Target: runs offline on a laptop CPU in under 4 GB of RAM.

> **Status:** working prototype, built for the [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475) (Oct 6–29, 2026). `generate → verify → train → evaluate` runs end to end on a laptop (RTX 2060 6 GB), including tool-calling traces against a mock Slurm cluster. Export (GGUF/Ollama) is not built yet.

## Quick start

```bash
uv sync --extra train              # plain `uv sync` removes torch/trl/peft
echo 'OPENCODE_API_KEY=...' > .env # teacher + judge endpoint (any OpenAI-compatible API works)
uv run distillkit generate -c configs/tools_pilot.yaml   # teacher writes questions, answers and tool-calling traces
uv run distillkit verify   -c configs/tools_pilot.yaml   # dedup, flag/bash checks, tool-call checks
uv run distillkit train    -c configs/tools_pilot.yaml   # LoRA / QDoRA SFT of the student
uv run distillkit evaluate -c configs/tools_pilot.yaml   # student vs. base: judge + tool-call scoring
uv run pytest -q
```

| Config | Use |
|---|---|
| `configs/tools_pilot.yaml` | small live pilot: prose + tool traces from the toy teacher API, Qwen3-0.6B student |
| `configs/toy_qdora.yaml` | local QDoRA smoke test (Qwen3-0.6B, 4-bit + DoRA) |
| `configs/qwen3_4b_qdora.yaml` | Leonardo: Qwen3-32B teacher on vLLM → Qwen3-4B-Instruct-2507, ~30% tool traces |
| `configs/accelerate/fsdp.yaml` | multi-GPU QDoRA: `accelerate launch --config_file configs/accelerate/fsdp.yaml -m distillkit.cli train -c <config>` (untested on Leonardo yet) |

Outputs go to the config's `run_dir` (git-ignored). Configs are validated on load (unknown keys are errors). API calls retry transient errors with exponential backoff. `generate` appends results as they finish, so rerunning it after a crash or a wall-time kill only does the missing work. With `teacher.top_logprobs: 20`, the teacher's top-20 logprobs for every prose answer token go to `run_dir/teacher_logprobs.jsonl.gz`, keyed by row `id`.

## Tool calling

The assistant has 8 Slurm tools (`src/distillkit/tools.py`): `job_status`, `list_queue`, `submit_job`, `cancel_job`, `read_job_log`, `job_accounting`, `gpu_availability`, `partition_info`. Calls use Qwen3's own Hermes format (`<tool_call>{"name": …, "arguments": …}</tool_call>`), which vLLM (`--tool-call-parser hermes`), llama-server (`--jinja`) and Ollama parse into OpenAI `tool_calls`, so no external agent framework is needed at runtime. For data generation and evaluation the tools run against a deterministic mock cluster. Generated traces are kept only if every call is valid and executable, every job id comes from the question or an earlier tool result, and calling (or not) fits the question.

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
- **[PROGRESS.md](PROGRESS.md):** running log of what is done.

## Planned stack

Python · PyTorch · Hugging Face Transformers / TRL / PEFT (DoRA) · bitsandbytes · vLLM · FSDP · llama.cpp (GGUF) · Ollama

## Contributing

Work happens on the `dev` branch; `main` holds reviewed milestones. Open a pull request against `dev`.

## License

[MIT](LICENSE) © 2026 DistillKit contributors
