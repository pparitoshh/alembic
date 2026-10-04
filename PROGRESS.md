# DistillKit — Progress Log

*Tracks what is done against [GOAL.md](GOAL.md) (order of work §6, timeline §11). Newest entries first.*

## Status at a glance

| Stage | Code | Ran locally | Notes |
|---|---|---|---|
| Seeds + doc-level split | ✅ | ✅ | 6 Slurm docs; `slurm_job_arrays` and `slurm_requeue_signals` held out for eval |
| Eval set (prose) | ✅ | — | 26 hand-written Qs from 2 held-out docs (target: ~50 from 3+ docs before Oct 6) |
| Eval set (tool slice) | ✅ | — | `data/eval/eval_tools.jsonl`: 16 draft items (10 call, 6 no-call), needs human review |
| Tools: 8 schemas + mock cluster | ✅ | ✅ | `tools.py`: deterministic, Leonardo-like partitions |
| Tool-call checker | ✅ | ✅ | `toolcheck.py`: schema, execution, job-id grounding, decision, BFCL-style AST |
| Gold examples | ✅ | ✅ | `data/gold/prose.json`, `tool_trace.json` |
| `generate` (prose + tool traces) | ✅ | ✅ | resumable; ~30% tool traces in the Leonardo config |
| `verify` | ✅ | ✅ | dedup, flag linter, `bash -n`, tool checks, decontamination |
| `train` (LoRA / QDoRA) | ✅ | ✅ | QDoRA on Qwen3-0.6B (RTX 2060); FSDP config written, untested |
| `evaluate` | ✅ | — | prose judge + tool-slice scoring; not yet run on a tool-trained student |
| Teacher top-20 logprob capture | ✅ | ✅ | works against the toy API; needs a vLLM/llama-server test on the cluster |
| `export` (GGUF/Ollama) | ✅ | ✅ | merge → bf16 GGUF → imatrix → Q4_K_M/Q8_0 → Modelfile → `ollama create` + tool-call smoke test |

## Log

### 2026-10-04: hermetic tests, portable package

- `tests/conftest.py` (autouse, every test): network blocked, `*_API_KEY`/`*_TOKEN` removed, Hugging Face offline, HOME and the working directory moved to a temp dir, module state reset. `tests/test_hermetic.py` checks these guarantees.
- `.env` is read from the working directory only. Before, `load_dotenv()` searched upward from `config.py` and loaded the repo's API key even in tests run elsewhere.
- The Slurm flag list ships inside the package (`src/distillkit/data/slurm_flags.txt`); `verify.flag_list` is optional and, when set, also drives the mock `sbatch`. Before, the mock found the list via a repo-relative path and silently skipped the check in an installed copy.
- `bash -n` fails with a clear error if bash is missing.
- **Checked:** 51 tests pass in the repo, and against a wheel installed non-editable in a fresh venv, run from `/tmp` with an empty environment (no torch, no HOME, no credentials), as after `pip install` on Leonardo.

### 2026-10-04: export stage (branch `feature/export`)

- `export.py`: merge the (Q)DoRA adapter into the bf16 base → `convert_hf_to_gguf.py` → `llama-imatrix` on the run's own verified transcripts (tools included) → `llama-quantize` per `export.quants` → Ollama `Modelfile` → optional `ollama create`. Each step is skipped if its output exists.
- Modelfile template: Ollama's own `qwen3` template (`ollama show qwen3:4b --modelfile`) minus thinking, so Ollama's tool-call parser works unchanged. The text after `<|im_start|>assistant` comes from the student's tokenizer (empty for 2507, an empty think block for hybrid Qwen3), matching training.
- llama.cpp b11392 from the release tarball (no compiler on this laptop) + the matching source tag for the converter. Converter deps (`sentencepiece`, `protobuf`, …) are in a new `export` extra; llama.cpp's own pins (older torch/transformers) aren't needed.
- **Pilot:** `tools_pilot` adapter (Qwen3-0.6B) → Q4_K_M 0.40 GB, Q8_0 0.64 GB in ~2 min. Through Ollama's chat API the model emitted a Hermes tool call that Ollama parsed into `job_status(job_id="4718207")`, then answered correctly from the mock result. Prose quality is still toy-level (`--gres=2`), expected at 0.6B × 15 examples.

### 2026-10-04: tool calling, outdated material removed

**Tools and checks**
- `tools.py`: 8 tools (`job_status`, `list_queue`, `submit_job`, `cancel_job`, `read_job_log`, `job_accounting`, `gpu_availability`, `partition_info`) as JSON schemas, with a deterministic mock cluster: the same job id always gives the same state, logs and usage. `submit_job` lints the script (flags, `bash -n`, partition) like `sbatch`.
- `toolcheck.py`: a trace passes when every call is valid and runs on the mock, **every `job_id` comes from the question or an earlier tool result** (never invented; checked in all modes), and the decision fits the mode (`call` must call, `none` must not; `ask` may ask back or look the job up). Eval scoring adds BFCL-style AST match (`""` = optional, `*` = any value, lists of alternatives) and a grounded-id rate.
- Tool-call format is Hermes (Qwen3's own template); tools are passed wrapped as `{"type": "function", ...}`, as vLLM, llama-server and Ollama do.

**Generation**
- Questions get a mode: `prose`, or for `generate.tool_fraction` of them `call` / `ask` / `none`. Tool modes run the teacher as an agent against the mock cluster (up to `max_tool_rounds`); prose ids don't change when `tool_fraction` does.
- Gold anchors go in the system prompt as a labelled example. As earlier chat turns they leaked: the teacher took the example's job id for the user's.
- Questions that mention a job get its real mock state, so the premise matches what the tools return.

**Live pilot** (`configs/tools_pilot.yaml`, toy teacher API): 15 questions (6 tool-mode), ~1.5 min; 15/15 kept after the fixes; LoRA training on the result works with the `tools` column. Earlier pilot rounds found and fixed: gold leakage, `submit_job` accepting unknown partitions, `--` read as an empty flag, and a too-strict "ask" rule (replaced by the job-id grounding rule).

**Removed:** the toy iteration configs (`toy.yaml`, `iter*.yaml`), their write-ups and report (`docs/`), the old training guide, and the references to them. They described the Qwen3-0.6B + plain LoRA toy setup.

**Student:** Qwen3-4B-Instruct-2507 (non-thinking, better tool use; GOAL.md §3).

### 2026-10-04: refactor for the final design

- **Typed config** (`config.py`, pydantic): unknown keys fail fast.
- **One record format** (`records.py`): rows are chat `messages` (+ optional `tools`), so prose answers and tool-call traces share one pipeline.
- **Teacher client** returns text, tool calls and optional top-k logprobs (`teacher.top_logprobs`); `extra_body` passes server options (vLLM `return_tokens_as_token_ids`, `enable_thinking`).
- **Resumable generation**: results are appended as they finish (`questions.jsonl`, `generated.jsonl`, `teacher_logprobs.jsonl.gz`) with stable ids; a rerun does only what's missing, and a line cut off by a kill is repaired.
- **Training**: conversational data with `assistant_only_loss` (checked by hand: loss covers assistant turns + `<|im_end|>`, not tool results; the prefix matches the inference prompt). `train.method: lora|dora`, `load_in_4bit` (NF4, `quant_storage` = model dtype for FSDP). TRL casts QLoRA adapters to bf16, which breaks fp16 AMP, so they're cast back to fp32 on the fp16 path.
- Configs: `qwen3_4b_qdora.yaml` (Leonardo), `toy_qdora.yaml` (local), `tools_pilot.yaml` (tool pilot), `accelerate/fsdp.yaml` (PEFT's FSDP + QLoRA recipe).

### 2026-10-04: goals finalized

- **GOAL.md** replaced with the Oct 4 version; **RESEARCH.md** revised for it (vLLM + `Qwen3-32B-AWQ` recommended for the teacher; top-k logprob caching is biased; QDoRA issues under ZeRO-2; Arcee AI already ships a "DistillKit"; ≥ 20 tok/s on a laptop CPU is a risk). `PLAN.md` and `GOAL_v2.md` removed.

## Next steps

1. Human review of the eval sets; grow the prose set to ~50 questions from 3+ held-out docs.
2. Laptop benchmarks for G3/G6: `llama-bench` tokens/s and peak RAM per quant level; accuracy per quant via the eval harness on the GGUF models.
3. Slurm job scripts for vLLM teacher serving and training on Leonardo; day-1 smoke test of logprobs with token IDs.
4. Baselines: Qwen3-4B-Instruct-2507 untrained on both eval slices.
