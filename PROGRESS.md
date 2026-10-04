# DistillKit — Progress Log

*Tracks what is done against [GOAL.md](GOAL.md) (order of work §6, timeline §11). Newest entries first.*

> **Note (Oct 4):** iterations 0–4 below are toy runs with **Qwen3-0.6B + plain LoRA** on a laptop. The final setup is **Qwen3-32B teacher → Qwen3-4B student with QDoRA + FSDP, plus tool calling** (see [GOAL.md](GOAL.md) and [RESEARCH.md](RESEARCH.md)). The old PLAN.md was removed; entries that mention it are kept as history.

## Status at a glance

| Stage | Code | Ran on toy data | Notes |
|---|---|---|---|
| Seeds + doc-level split | ✅ | ✅ | 6 Slurm docs; `slurm_job_arrays` and `slurm_requeue_signals` held out for eval |
| Eval set | ✅ | — | 26 hand-written Qs from 2 held-out docs (target: ~50 from 3+ docs before Oct 6) |
| `generate` | ✅ | ✅ | 60 questions × 2 answers = 120 rows (12 min at concurrency 12); now with short-answer prompt and tenacity retries |
| `verify` | ✅ | ✅ | 120/120 kept (40/40 earlier, after fixing checker false positives) |
| `train` (LoRA) | ✅ | ✅ | Qwen3-0.6B, fp16; best setting lr 1e-4, r16/α32, 3 epochs (24 steps, loss ≈ 1.39); `train.seed` option |
| `evaluate` | ✅ | ✅ | student **beats base on average**: 64.1% ± 4.8 over 3 seeds, 26 questions (range 59.6–69.2%) ([report](docs/report_iterations_0-4.md)) |
| `export` (GGUF/Ollama) | ❌ | ❌ | not started |
| Tool calling (schemas, mock tools, traces, checker) | 🟡 | ❌ | data format + training mask ready; schemas, mock tools, generation and checker still to do |
| Teacher top-20 logprob capture | ✅ | ❌ | `teacher.top_logprobs`; needs a vLLM/llama-server test on the cluster |
| QDoRA + FSDP training | ✅ | ✅ QDoRA (1 GPU) | `train.method: dora`, `load_in_4bit`; FSDP config written, untested |

## Log

### 2026-10-04: refactor for the final design (branch `refactor/qdora-tool-ready`)

**Kept as is:** `seeds.py` (doc-level split, chunking), `checks.py` (flag linter, `bash -n`), dedup/decontamination in `verify.py`, the both-orders judge in `evaluate.py`, tenacity retries, `schemas.py`.

**Changed**
- **Typed config** (`config.py`, pydantic): unknown keys fail fast; new fields have defaults, so every old config still loads (a test covers this).
- **One record format** (`records.py`): rows are chat `messages` (+ optional `tools`), so prose answers and tool-call traces share one pipeline. Old `question`/`answer` rows still load.
- **Teacher client** returns text, tool calls and optional top-k logprobs (`teacher.top_logprobs`); `extra_body` passes server options (vLLM `return_tokens_as_token_ids`, `enable_thinking`).
- **Resumable generation**: questions and answers are appended as they finish (`questions.jsonl`, `generated.jsonl`, `teacher_logprobs.jsonl.gz`) with stable ids; a rerun does only what's missing, and a line cut off by a kill is repaired.
- **Training**: conversational data with `assistant_only_loss` (TRL's Qwen3 training template). The mask was checked by hand: loss covers assistant turns + `<|im_end|>`, not tool results, and the prefix matches the inference prompt. `train.method: lora|dora`, `load_in_4bit` (NF4, `quant_storage` = model dtype for FSDP), `lora_dropout`.
- New configs: `qwen3_4b_qdora.yaml` (Leonardo template), `toy_qdora.yaml` (local), `accelerate/fsdp.yaml` (PEFT's FSDP + QLoRA recipe).
- CLI stages are a registry, so `export` plugs in as one entry.

**Verified:** 26 tests pass. Local QDoRA run on Qwen3-0.6B (RTX 2060, fp16): 120 examples × 3 epochs in 2m15s, DoRA adapter saved, eval generation loads it. One fix was needed: TRL casts QLoRA adapters to bf16, which breaks fp16 AMP, so they're cast back to fp32 on the fp16 path.

**Not done (features, not refactor):** tool schemas + mock tools + trace generation + tool-call checker; `export` stage; judge hosted on the cluster; FSDP run on real multi-GPU hardware.

### 2026-10-04: goals finalized, docs rewritten

- **GOAL.md** replaced with the Oct 4 version: Qwen3-32B (4-bit) teacher on one A100 64 GB, Qwen3-4B student with QDoRA + FSDP, tool calling (~30% of data), top-20 teacher logprobs saved for v2 logit KD, one shared Leonardo node, eval-first order of work.
- **RESEARCH.md** revised for this design. Key findings: vLLM GGUF is "highly experimental" → recommend vLLM + official `Qwen3-32B-AWQ` (native top-20 logprobs with token IDs); consider `Qwen3-4B-Instruct-2507` as the student (BFCL-v3 61.9 vs 57.6); top-k logprob caching is biased (Sparse Logit Sampling, ACL 2025); QDoRA has known issues under DeepSpeed ZeRO-2 and is ~1.5–1.8× slower than LoRA; **Arcee AI already ships a "DistillKit"**, so the name needs a decision before release; ≥ 20 tok/s on a laptop CPU for 4B Q4_K_M is a risk.
- Removed the outdated `PLAN.md` (1.7B LoRA, 2 nodes) and `GOAL_v2.md` draft.

**Next:** eval set (~50 Qs incl. tool slice), 5–10 tool schemas + mock tools, 2 gold examples, export stage, Slurm scripts, day-1 teacher logprob smoke test.


### 2026-09-30 (night): iterations 1-4, goal met on average

Full write-up: [docs/report_iterations_0-4.md](docs/report_iterations_0-4.md); per-iteration notes in `docs/iter_1_learning.md` to `iter_3_learning.md`. Each experiment ran in a git worktree with its own config and `run_dir`.

**Done**

| Iter | Change | Win rate vs base |
|---|---|---|
| 1 | lr 2e-4 → 5e-5, LoRA r 64/α128 → 16/32 | 45.0% (looping 9/10 → 1/10) |
| 2 | 3× data (120 samples), "short and direct" teacher prompt | 42.5% (no real change) |
| 3 | lr 1e-4, 3 epochs (24 steps) | 77.5% (10 questions, 1 seed) |
| 4 | seed test: seeds 42/1/2, 26 questions incl. a new held-out doc | **64.1% ± 4.8** (69.2 / 63.5 / 59.6) |

- Added `train.seed` config option (default 42), second held-out doc `slurm_requeue_signals` with 16 eval questions (`data/eval/eval_req.jsonl`, combined `eval_all.jsonl`).
- **API retries with tenacity** (exponential backoff + jitter, 8 attempts) after the judge endpoint returned `429: Endpoint is unavailable` and crashed two evaluations (re-run). `tenacity` added as a dependency.
- Pushed `dev` to origin.

**Findings**
- Looping was caused by over-aggressive training (iteration 0), not the data.
- The one change that clearly helped: more training signal at a moderate lr (1e-4, 3 epochs). Data volume alone did nothing. lr and epochs were changed together, so they are not separated yet.
- The first 77.5% was optimistic. Seed spread is about ±5 points; on the new held-out doc the mean is 59.9%, right at the target line. Differences under ~10 points between settings cannot be trusted with this eval.
- The student makes fewer invented flags than the base (≈0.066 vs 0.143 of flags used) but script answers are still often wrong (e.g. for-loop instead of `--array`).
- **Gotcha:** plain `uv sync` removes the `train` extra (torch etc.) from the venv and silently broke the first seed runs. Use `uv sync --extra train`.

**Next**
- Ablate lr and epochs separately (3 seeds each on `eval_all`), grow the eval set to 50+ questions across 3+ docs, add more script/`--array` training examples, rebuild `slurm_flags.txt` from man pages.

### 2026-09-30 (evening): first `evaluate`

Full write-up: [docs/iter_0_learning.md](docs/iter_0_learning.md).

**Done**
- Judge fixes: verdict was read from the reply's *first* character → now the last standalone A/B/T, raw replies saved, `judge_unparsed` count added. First run had 13/20 empty replies (reasoning hit `max_tokens: 512`) → raised to 4096.
- Judge calls now run in parallel (`Teacher.map`, concurrency 4). Full evaluate: 3m11s.

**Result**
- Student vs base win rate **27.5%** (target > 60%). The student repeats itself and hits the 512-token limit in 9/10 answers (base: 0/10), even on its own training questions.
- Training data checked and correct (prompt masked, stop token trained on). Likely cause: training too aggressive for 0.6B on 40 samples (lr 2e-4, LoRA r=64) and long teacher answers (median 245 tokens). Not yet tested.

**Next**
- Retrain with lr 5e-5, r=16 into a separate folder; check answer lengths before any judge run.

### 2026-09-30 (later)

**Done**
- **First `train` run** (Qwen3-0.6B, LoRA r=64 on all 7 linear layer types, fp16): 40 samples × 2 epochs = 6 optimizer steps, 28 s of training. Loss 1.85 → 1.33, token accuracy 59% → 69%. Adapter saved to `runs/toy/adapter` (162 MB, fp32; 40.4M trainable params = 6.3%). On 40 samples this is mostly memorisation, so it only proves the pipeline works.
- Two fixes were needed to get there:
  - transformers 5 dropped `warmup_ratio` → `warmup_steps=0.03` (a float < 1 is read as a ratio).
  - torch 2.14 routes RoPE's `bmm` to a Triton kernel that needs a C compiler (none installed) → `cli.py` sets `TORCH_DISABLE_NATIVE_JIT=1`.
- `docs/training.md`: plain-language onboarding guide to training (one sample, next-token prediction, micro-batches vs optimizer steps, frozen vs LoRA weights).
- **First `verify` run:** initially kept 36/40. All 4 rejections were false positives in the checker, not bad answers:
  - 2 × `bad_flags`: `srun --jobid=...` is a real flag but was missing from the hand-written `slurm_flags.txt` → added `jobid`.
  - 2 × `bash_syntax`: doc-style placeholders like `<jobid>` parse as redirections under `bash -n` → `bash_syntax_ok` now replaces `<placeholder>` tokens before checking (new test added; 5 tests pass).
- After the fixes: **40/40 kept** in `runs/toy/verified.jsonl`.

**Findings / issues**
- The hand-written flag list will keep rejecting real-but-rare flags until it is rebuilt from the real man pages (next step 6).

### 2026-09-30

**Done**
- `PLAN.md` written. Safest route: Qwen3-1.7B + LoRA, Slurm-first, eval before training, ~5k verified examples.
- Project scaffolded with a **uv-managed `.venv`** (torch 2.14+cu130, transformers 5.17, TRL 1.14, PEFT 0.21; CUDA works on the RTX 2060).
- Package `src/distillkit/` with the CLI `uv run distillkit {generate,verify,train,evaluate,all}`.
- Checkers (`checks.py`) with 4 passing unit tests: `#SBATCH`/`srun`/`sbatch` flag extraction, hallucinated-flag detection, `bash -n`.
- Teacher switched from Ollama to **OpenCode Zen** (OpenAI-compatible `https://opencode.ai/zen/v1`):
  - teacher `qwen3.8-flash` ($0.15 / $0.47 per 1M tokens)
  - judge `deepseek-v4.1-flash` (different model family from the teacher)
  - API key read from `.env` (`OPENCODE_API_KEY`, gitignored); `.env.example` committed.
- **First `generate` run:** 40 rows, 0 empty answers, average 890 characters per answer. Spot check: answers are grounded, use fenced scripts, and state their assumptions.

**Findings / issues**
- `qwen3.8-max` and `deepseek-v4-flash` returned 403 "Model access is disabled" on our key. The config uses models that were verified to work.
- The judge model reasons before answering (~50 tokens); `max_tokens: 16` returned empty text, so it is now 512.
- **Generation is slow: 7m38s for 60 calls** (~30 s per call at concurrency 4). This must be fixed before the 5k pilot (see next steps).
- Generated questions are long and very specific. Fine for now, but we need shorter, casual questions too (add a "question length/format" dimension to the grid).
- One answer contained a wrong command (`python train.py --nproc_per_node=8`). The current flag checker only looks at Slurm commands, so only the judge would catch this.

## Next steps

1. ~~**`verify` on the toy data**: check rejection reasons in `runs/toy/rejected.jsonl` for false positives.~~ Done: 40/40 kept after checker fixes.
2. ~~**`train`**: LoRA on Qwen3-0.6B locally; confirm loss decreases and the adapter saves.~~ Done: loss 1.85 → 1.33, adapter saved.
3. ~~**`evaluate`**: base vs. student on the 10 eval Qs; confirm the judge verdict parsing works.~~ Done: runs, but student loses (27.5%) due to looping. Fix before scaling: gentler LoRA settings, shorter teacher answers (see [iter 0](docs/iter_0_learning.md)).
4. **Speed up generation**: concurrency 12 still took 12 min for 120 answers; measure per-call latency, try 16+, and log tokens/cost per run.
5. **`export`**: merge LoRA → GGUF Q4_K_M → Ollama `Modelfile`.
6. **Real data (by Oct 5)**:
   - replace the hand-written seeds with real Slurm man pages/docs (record licenses)
   - extract `slurm_flags.txt` from real `man sbatch`/`srun`/`salloc`
   - grow the eval set from 26 to ~50 questions across 3+ held-out docs
7. Add question-format diversity (short / long-with-logs) to the generation grid.
8. **Ablate lr vs. epochs** (lr 1e-4 with 2 epochs; lr 5e-5 with 4-5 epochs), 3 seeds each.
9. **Fix script answers** (e.g. `--array` misuse): more `script`/`howto` training examples, stricter verification of teacher scripts.
