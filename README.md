# DistillKit

**Modular teacher–student distillation for local, domain-specific assistants.**

DistillKit turns a large open-weight **teacher** model into a compact **student** that runs on a laptop. The student learns from teacher-generated question/answer text and tool-calling traces (sequence-level distillation), so any teacher that emits text works with any student. The teacher's top-20 logprobs are saved as well, so logit-level distillation can be added later without re-running the teacher.

The demo use case is an **HPC Assistant**: Qwen3-32B distilled into a **Qwen3-4B-Instruct-2507** student (QDoRA) specialised in Slurm, CUDA, MPI and profiling, with tool calling (job status, submission, logs, GPU availability). Target: runs offline on a laptop CPU in under 4 GB of RAM, as a streaming chat assistant (≥ 8 tok/s with 2,048 tokens of context; ≤ 3 s to the first token on follow-up questions).

> **Status:** working prototype, built for the [European AI Hackathon](https://www.openhackathons.org/s/siteevent/a0CUP00003yKxcX2AS/se000475) (Oct 6–29, 2026). `generate → verify → train → evaluate → export` runs end to end on a laptop (RTX 2060 6 GB), including tool-calling traces against a mock Slurm cluster and an Ollama package whose tool calls Ollama parses. Slurm scripts for Leonardo are ready; laptop speed/RAM benchmarks of the 4B are in [research/](research/README.md).

## Quick start

```bash
uv sync --extra train --extra export   # plain `uv sync` removes torch/trl/peft
echo 'OPENCODE_API_KEY=...' > .env # teacher + judge endpoint (any OpenAI-compatible API works)
uv run distillkit generate -c configs/tools_pilot.yaml   # teacher writes questions, answers and tool-calling traces
uv run distillkit verify   -c configs/tools_pilot.yaml   # dedup, flag/bash checks, tool-call checks
uv run distillkit train    -c configs/tools_pilot.yaml   # LoRA / QDoRA SFT of the student
uv run distillkit evaluate -c configs/tools_pilot.yaml   # student vs. base: judge + tool-call scoring
uv run distillkit export   -c configs/tools_pilot.yaml   # merge -> GGUF (imatrix) -> Q4_K_M/Q8_0 -> Ollama
uv run distillkit bench    -c configs/tools_pilot.yaml   # laptop speed, peak RAM, time to first token per quant
uv run pytest -q
```

| Config | Use |
|---|---|
| `configs/tools_pilot.yaml` | small live pilot: prose + tool traces from the toy teacher API, Qwen3-0.6B student |
| `configs/toy_qdora.yaml` | local QDoRA smoke test (Qwen3-0.6B, 4-bit + DoRA) |
| `configs/qwen3_4b_qdora.yaml` | Leonardo: Qwen3-32B-AWQ teacher on vLLM → Qwen3-4B-Instruct-2507, ~30% tool traces, gpt-oss-20b judge |
| `configs/accelerate/fsdp.yaml` | multi-GPU QDoRA: `accelerate launch --config_file configs/accelerate/fsdp.yaml -m distillkit.cli train -c <config>` (untested on Leonardo yet) |

Outputs go to the config's `run_dir` (git-ignored). Configs are validated on load (unknown keys are errors). API calls retry transient errors with exponential backoff. `generate` appends results as they finish, so rerunning it after a crash or a wall-time kill only does the missing work. With `teacher.top_logprobs: 20`, the teacher's top-20 logprobs for every prose answer token go to `run_dir/teacher_logprobs.jsonl.gz`, keyed by row `id`.

## Models (Leonardo)

| Role | Model | Disk | GPU memory | Served by |
|---|---|---|---|---|
| Teacher | `Qwen/Qwen3-32B-AWQ` | ~19 GB | ~19 GB + KV cache, 1× A100 64 GB | vLLM (fallback: llama.cpp + GGUF Q4_K_M) |
| Student | `Qwen/Qwen3-4B-Instruct-2507` | ~8 GB | 4-bit QDoRA, 1–2× A100 (FSDP) | TRL; vLLM for eval answers |
| Judge | `openai/gpt-oss-20b` | ~14 GB | ~16 GB (MXFP4), 1× A100 | vLLM |
| Cross-check judge | `google/gemma-4-26B-A4B-it` | ~52 GB | ~52 GB (bf16), 1× A100, little room for KV cache | vLLM |

Both judges come from different model families than the Qwen teacher and student, so they don't favour their answers, and they run on the cluster, so evaluation needs no external API. The judge compares two answers pairwise, in both orders, against the reference chunk. The cross-check judge re-scores the same cached answers (`--set eval.tag=gemma4 --set judge.model=google/gemma-4-26B-A4B-it`), and agreement between the two makes "beats base" credible. Each judge is first validated with `distillkit calibrate` on known-label pairs (`data/eval/judge_calibration.jsonl`; results in [research/judge_calibration/](research/judge_calibration/2026-10-04_opencode/README.md)). Model sizes and the `$WORK` storage plan are in [GOAL.md](GOAL.md) §4.

## Export (GGUF + Ollama)

`export` merges the adapter into the full-precision base, converts it with llama.cpp, computes an importance matrix from the run's own training transcripts, quantizes to `export.quants`, and writes an Ollama `Modelfile`. The Modelfile uses Ollama's own Qwen3 tool-calling template, with the generation prompt taken from the student's tokenizer so it matches training. With `export.ollama_name` set, it also runs `ollama create` and a tool-call smoke test. Everything lands in `run_dir/export/`, and finished steps are skipped on a rerun.

llama.cpp setup (no compiler needed): a checkout for the converter, plus the release binaries in `bin/`:

```bash
git clone --depth 1 --branch b11392 https://github.com/ggml-org/llama.cpp ~/tools/llama.cpp
mkdir -p ~/tools/llama.cpp/bin
curl -L https://github.com/ggml-org/llama.cpp/releases/download/b11392/llama-b11392-bin-ubuntu-x64.tar.gz \
  | tar -xz -C ~/tools/llama.cpp/bin --strip-components=1
```

Pilot result (Qwen3-0.6B): Q4_K_M 0.40 GB, Q8_0 0.64 GB, about 2 minutes on a laptop CPU.

## Laptop benchmark and GGUF eval

`bench` measures every exported quant on the CPU: llama-bench prompt and generation speed with an
empty context and with 2,048 tokens in it, peak RAM at the deployed context (8,192, KV cache
included), and time to first token of a streaming chat on llama-server (a cold first question with
the system prompt and tool schemas, then a follow-up that reuses the processed prompt). It runs
`export.bench_runs` interleaved passes and reports mean ± std plus the machine's load, because a real
laptop is never idle. Results go to `run_dir/export/bench.{json,md}`.

Base Qwen3-4B-Instruct-2507 on an i7-9750H laptop (6 threads, 3 passes,
[full report](research/bench/2026-10-04_qwen3-4b-instruct-2507_base/README.md)):

| Quant | Peak RAM GiB (default / `--load-mode none`) | Gen tok/s, empty context | Gen tok/s, 2,048 in context |
|---|---|---|---|
| Q3_K_M | 3.81 / 3.13 | 12.6 | 7.6 |
| Q4_K_M | 5.15 / **3.52** | 11.3 | 7.2 |
| Q5_K_M | 3.88 | 9.6 | 6.5 |
| Q8_0 | 5.18 / 5.17 | 7.0 | 5.2 |

On AVX2 CPUs llama.cpp keeps a repacked copy of Q4_K/Q3_K weights; with the default memory-mapped
load the file stays resident too, so load Q4_K_M without mmap to stay under 4 GB.

`eval.gguf: [Q4_K_M, ...]` makes `answer`/`evaluate` also answer the eval set with the exported GGUF
files on llama-server (same rendered prompt, greedy), so the scores describe the file users run. The
judge compares each quant with the full-precision model it came from (0.5 = no quantization loss) and
with base.

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
- **[slurm/README.md](slurm/README.md):** running on Leonardo: setup, day-1 smoke tests, the job pipeline.
- **[research/](research/README.md):** committed results with the machine they ran on: laptop config, judge calibration, laptop benchmarks.
- **[PROGRESS.md](PROGRESS.md):** running log of what is done.

## Planned stack

Python · PyTorch · Hugging Face Transformers / TRL / PEFT (DoRA) · bitsandbytes · vLLM · FSDP · llama.cpp (GGUF) · Ollama

## Contributing

Work happens on the `dev` branch; `main` holds reviewed milestones. Open a pull request against `dev`.

## License

[MIT](LICENSE) © 2026 DistillKit contributors
