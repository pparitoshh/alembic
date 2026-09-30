# DistillKit — Progress Log

*Tracks what is done against [PLAN.md](PLAN.md). Newest entries first.*

## Status at a glance

| Stage | Code | Ran on toy data | Notes |
|---|---|---|---|
| Seeds + doc-level split | ✅ | ✅ | 5 Slurm docs; `slurm_job_arrays` held out for eval |
| Eval set | ✅ | — | 10 hand-written Qs (target: ~50 before Oct 6) |
| `generate` | ✅ | ✅ | 20 questions × 2 answers = 40 rows |
| `verify` | ✅ | ✅ | 40/40 kept after fixing 2 checker false positives |
| `train` (LoRA) | ✅ | ✅ | Qwen3-0.6B, fp16; 6 steps, loss 1.85 → 1.33 |
| `evaluate` | ✅ | ✅ | student **loses**: 27.5% win rate, loops in 9/10 answers ([iter 0](docs/iter_0_learning.md)) |
| `export` (GGUF/Ollama) | ❌ | ❌ | planned Oct 4 |

## Log

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
4. **Speed up generation**: raise `concurrency` (try 16), measure per-call latency, and log tokens/cost per run.
5. **`export`**: merge LoRA → GGUF Q4_K_M → Ollama `Modelfile`.
6. **Real data (by Oct 5)**:
   - replace the hand-written seeds with real Slurm man pages/docs (record licenses)
   - extract `slurm_flags.txt` from real `man sbatch`/`srun`/`salloc`
   - write ~50 eval questions from the held-out docs
7. Add question-format diversity (short / long-with-logs) to the generation grid.
