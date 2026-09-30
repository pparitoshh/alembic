# DistillKit — Execution Plan (safest route to "beats the baseline")

*Companion to [GOAL.md](GOAL.md) and [RESEARCH.md](RESEARCH.md). This file is the "what we actually do, in what order" plan.*

## 1. The one result we must get

> **A LoRA-fine-tuned Qwen3-1.7B, trained on ~5k teacher-generated, verified HPC Q&A pairs, beats its own base model on a held-out eval set** — pairwise LLM-judge win rate > 60% and fewer hallucinated `#SBATCH` flags — **and runs in Ollama as a Q4_K_M GGUF.**

Everything else in GOAL.md (multi-node, 50k examples, Pareto frontier, RAFT, DPO, second model family) is layered on top *after* this is green.

## 2. Safety decisions (and why)

| Decision | Choice | Why it's the safe option |
|---|---|---|
| Student | **Qwen3-1.7B** (hybrid, thinking **off**); **Qwen3-0.6B** for local dev | Mature in TRL, PEFT, llama.cpp, Ollama. Qwen3.5 is a Tier-2 bonus only. |
| Training method | **LoRA** (r=64, all linear layers, lr 2e-4), single GPU, TRL `SFTTrainer` | Fits on one GPU, no FSDP debugging, hard to break the model, matches full FT for SFT at this scale. |
| Teacher API | **One OpenAI-compatible client** — Ollama locally, `vllm serve` on the cluster | Same code path on laptop and cluster; swapping teachers is a config change. |
| Domain scope | **Slurm first**, then MPI/CUDA | Slurm is fully checkable offline (flag linter vs. man page) and the base model is weak at it → largest, most measurable gain. |
| Answers | Grounded in the source chunk, thinking off, short & direct, 4 answers/question | Least teacher hallucination; small students learn short answers best. |
| Data size | **~5k verified examples** for the first result | HPC-Coder-V2 saw plateaus at ~6k/subtype. Scale later only if time allows. |
| Order of work | **Eval set + baseline numbers before any training** | We can't claim "beats baseline" without a frozen, leak-free eval. |

## 3. Metrics (fixed before training)

1. **Pairwise judge win rate** — student vs. base, judge sees the reference chunk; both orders to cancel position bias.
2. **Flag hallucination rate** — share of `#SBATCH --flag` / `srun --flag` in answers that don't exist in the Slurm man pages (fully automatic).
3. **Script validity** — `bash -n` + static `#SBATCH` linter pass rate on "write a script" questions.
4. (Tier 1) `nvcc -c` / `mpicc` compile pass rate for CUDA/MPI questions.

## 4. Tiers

**Tier 0 — must (target: green by Oct 16, the GOAL.md checkpoint)**
- Toy pipeline runs end-to-end locally: seeds → generate → verify → train → eval.
- Eval set: ~50 hand-written + ~150 generated-then-human-reviewed questions from *held-out docs*.
- Baseline measured: Qwen3-1.7B base, zero-shot.
- ~5k verified Slurm/HPC examples from a 27B–35B teacher on the cluster.
- LoRA student trained, evaluated, beats base. Merged → GGUF Q4_K_M → Ollama.

**Tier 1 — should**
- Add MPI + CUDA topics with compile checks.
- Ablation A1: filtered vs. unfiltered (same size).
- Q8_0 vs. Q4_K_M eval; 0.6B vs. 1.7B student.
- Laptop RAM/tokens-per-second numbers.

**Tier 2 — stretch**
- Local RAG + RAFT; multi-node training; data-scaling curve; Qwen3.5/Gemma student; DPO.

## 5. Local dev setup (this laptop)

- RTX 2060, 6 GB, **no bf16** → fp16, Qwen3-0.6B for local training runs.
- Toy teacher: `qwen3.6-plus` via OpenCode Zen (`https://opencode.ai/zen/v1`, key in `$OPENCODE_API_KEY`); judge: `deepseek-v4.1-flash` (different family). Ollama remains an offline fallback.
- Python: **uv-managed `.venv` only** (`uv sync`, `uv run ...`). No system packages.

## 6. Pre-event schedule (Sep 30 – Oct 5)

| Day | Task | Done when |
|---|---|---|
| Sep 30 | Scaffold package, uv venv, toy configs, 5 seed chunks, 10 eval Qs | `uv run distillkit --help` works |
| Oct 1 | `generate` + `verify` (dedup, flag linter, `bash -n`) against Ollama teacher | toy `train.jsonl` with ~100 verified rows |
| Oct 2 | `train` (LoRA, Qwen3-0.6B, fp16) on toy data | loss goes down, adapter saved |
| Oct 3 | `evaluate` (flag check + pairwise judge) base vs. student | a results table prints |
| Oct 4 | `export`: merge → GGUF Q4_K_M → Ollama `Modelfile` | `ollama run distillkit-toy` answers |
| Oct 5 | Collect real seed corpus (Slurm man pages first), document-level split, write ~50 eval Qs | frozen `eval_core.jsonl` |

## 7. Hackathon weeks

| When | Task |
|---|---|
| Oct 6–9 | Cluster access; `vllm serve` teacher; smoke test Qwen3-1.7B LoRA on cluster GPU; **baseline numbers** |
| Oct 10–12 | 5k pilot generation (4 answers/q) → verify → first real student |
| Oct 13–16 | Fix what the first eval shows; **Tier 0 green** |
| Oct 17–26 | Tier 1 (MPI/CUDA, filtering ablation, quant levels), then Tier 2 if on track |
| Oct 27–29 | Freeze, model card, demo, report (including negative results) |

## 8. Fallbacks

| If… | Then… |
|---|---|
| Student doesn't beat base on judge | Check data quality by hand; lower LR/epochs (overfitting) or raise answers/question; add RAG to both and report student+RAG vs base+RAG |
| Qwen3-1.7B won't fit / train | Qwen3-0.6B (same recipe) |
| GGUF conversion fails | Use the pre-merged HF model + `llama.cpp` from a pinned release; worst case demo via HF transformers |
| Teacher too slow | Smaller/sparse teacher (e.g. 30B-A3B MoE), fewer answers per question |
