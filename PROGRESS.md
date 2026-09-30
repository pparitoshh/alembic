# DistillKit — Progress Log

*Tracks what is done against [PLAN.md](PLAN.md). Newest entries first.*

## Status at a glance

| Stage | Code | Ran on toy data | Notes |
|---|---|---|---|
| Seeds + doc-level split | ✅ | ✅ | 5 Slurm docs; `slurm_job_arrays` held out for eval |
| Eval set | ✅ | — | 10 hand-written Qs (target: ~50 before Oct 6) |
| `generate` | ✅ | ✅ | 20 questions × 2 answers = 40 rows |
| `verify` | ✅ | ✅ | 40/40 kept after fixing 2 checker false positives |
| `train` (LoRA) | ✅ | ⏳ | Qwen3-0.6B, fp16 (RTX 2060) |
| `evaluate` | ✅ | ⏳ | flag hallucination + pairwise judge (both orders) |
| `export` (GGUF/Ollama) | ❌ | ❌ | planned Oct 4 |

## Log

### 2026-09-30 (later)

**Done**
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
2. **`train`**: LoRA on Qwen3-0.6B locally; confirm loss decreases and the adapter saves.
3. **`evaluate`**: base vs. student on the 10 eval Qs; confirm the judge verdict parsing works.
4. **Speed up generation**: raise `concurrency` (try 16), measure per-call latency, and log tokens/cost per run.
5. **`export`**: merge LoRA → GGUF Q4_K_M → Ollama `Modelfile`.
6. **Real data (by Oct 5)**:
   - replace the hand-written seeds with real Slurm man pages/docs (record licenses)
   - extract `slurm_flags.txt` from real `man sbatch`/`srun`/`salloc`
   - write ~50 eval questions from the held-out docs
7. Add question-format diversity (short / long-with-logs) to the generation grid.
