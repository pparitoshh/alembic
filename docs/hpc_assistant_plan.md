# HPC Assistant: plan

*Oct 18, 2026*

## Goal

We're building a generic HPC assistant that runs fully on a laptop. A user installs one thing and asks questions about Slurm, job scripts, modules, storage, GPUs or any HPC site. Users can also load their own cluster manual. Documentation search is built in, so nobody has to set up RAG.

## Direction: bring your own data

HPC is the first use case for a generic local expert assistant: strong tool calling plus support for external documents, so anyone can bring their own data (food, budget or diet planning, and so on) and get an expert assistant. To keep that path open:
- The assistant core (`hpc-assistant`: document loading, search, the `search_docs` tool, chat) has no HPC-specific logic except the built-in index and the system prompt. Both can be swapped.
- The training recipe (verified, source-grounded Q&A generated from a domain's documents, RAFT with and without context) works for any document set.
- A separate repository for the generic assistant can come later. This branch keeps the HPC version working end to end first.

## What the user gets

```
hpc-assistant "How do I request 2 GPUs for 4 hours with Slurm?"
hpc-assistant --add our-cluster-manual.pdf     # index your own manual, then chat
hpc-assistant                                   # chat with history: /add FILE, /docs on|off, /reset, /quit
hpc-assistant --no-docs "..."                   # answer from the model alone
```

- Answers stream like a normal chat assistant, with the sources listed underneath.
- Everything stays local: the model, the search index and the user's documents.

## How it works

| Part | What it is |
|---|---|
| Model | Qwen3-4B-Instruct-2507 fine-tuned on our verified HPC Q&A, merged and quantized to GGUF Q4_K_M (about 2.5 GB). It runs under Ollama or llama.cpp. |
| Built-in docs | Two SQLite FTS5 files (about 220 MB): generic Slurm/LUMI/CINECA docs and all European AI Factory docs (7,159 docs, 48K chunks). |
| User docs | `--add` indexes PDF, Markdown, text or HTML into a local `user_docs.sqlite`. Re-adding a file replaces it. |
| Retrieval (automatic) | Every question searches the user docs first, then the built-in docs. The top 3 excerpts go into the prompt in the exact format the model was trained and evaluated with. |
| Generic fallback | Search is matched to the site named in the question (LUMI, Leonardo, the AI Factories, generic Slurm). If that finds nothing, the assistant searches all docs regardless of site, so questions about any other cluster still get context. |
| Retriever as a tool | The model can call `search_docs(query)` during an answer for follow-up lookups. |
| No docs needed | The model also answers without retrieval. Training removes the excerpt from about 35% of the AI Factory questions, so it learns the facts itself. |

Cost of built-in retrieval, measured on Leonardo:
- Search takes about 6 ms per question.
- The real cost is prompt length: about 1,900 extra tokens per question.
- That is why we also train the model to answer well without docs, and why `--no-docs` and fewer excerpts are options.

## Training data

Every training row is verified: an independent reviewer model checks it against its source, then deterministic checks run (commands, flags, scripts), then rows overlapping the evaluation sets are removed and near-duplicates are dropped.

| Source | Verified Q&A |
|---|---|
| Round-3 set (B7 answers regenerated at full length, plus CC-BY rows) | 19,411 |
| AI Factory batch 1 (all European AI Factory sources) | ~18,100 now, ~23K at the end |
| AI Factory top-up (new question angles, more ask-back questions) | ~3,400 now, ~15K expected |
| **Total** | **~41K now; ~43K in round 4; 60K+ in round 5** |

Every row carries its source licence. A public dataset release would ship only the permissive subset (`verified_permissive.jsonl`).

## Training and evaluation rounds

| Round | Data | Status |
|---|---|---|
| 3 | 19.4K rows | Done. Adapter ×0.5 + RAG: v3 **0.682** / v2 **0.635** / d1 **0.554**, against base + RAG 0.640 / 0.545 / 0.493 in the same judge run |
| 4 | ~43K rows (round 3 + batch 1) | Starts about 14:30 CEST Oct 18, scores about 22:00 |
| 5 | 60K+ rows (round 4 + top-up), with 35% of factory rows trained without docs | Overnight, scores Oct 19 morning |

- Each round trains 3 seeds and tests the adapter at strength ×1.0, ×0.5 and ×0.2, with docs and without.
- Earlier rounds are re-judged in the same judge run, so the round-to-round comparisons (paired bootstrap) aren't affected by judge drift between runs.
- The judge is gpt-oss-20b, both answer orders, calibrated at 0.94 accuracy.

## Packaging (GOAL.md G3)

1. Merge the winning adapter into the base model, convert to GGUF, quantize to Q4_K_M, and build an Ollama Modelfile. The existing `distillkit export` stage does this.
2. Ship three things: the GGUF model, the two index files and the `hpc-assistant` command (Python standard library only; `pypdf` optional for PDFs).
3. Check on a CPU-only machine: memory below 4 GB, streaming at reading speed, timed answers with and without docs. A laptop-like test on Leonardo CPU is running now.

## Done when

- [ ] Final model beats base + RAG on v3, v2 and d1 in the same judge run (round 3 already does)
- [ ] The no-docs score improves over round 4 (round 5's goal)
- [ ] Runs under 4 GB RAM on CPU and streams at reading speed
- [ ] `--add manual.pdf` works on a real cluster manual
- [ ] Model card with data sources, licences and evaluation

## Open points

1. **Licence of the built-in index.** It contains the text of all AI Factory pages, and many have unspecified or NC-SA licences. A public package may only bundle the permissive subset; the full index stays an internal or download-it-yourself option. This needs a team decision before release.
2. **The search tool is untrained.** The model was trained on tool calls in this format, but not on `search_docs` itself. Automatic retrieval works regardless; if the tool proves useful, we can add training rows for it.
3. **Laptop speed with docs.** About 1,900 extra prompt tokens costs a few seconds on a CPU. Options: fewer excerpts, shorter excerpts, or relying more on the no-docs ability trained in round 5.
4. **PDF quality.** Scanned PDFs have no text layer and need OCR, which is not planned.

## Code

- `src/distillkit/assistant.py`: the assistant (retrieval, user docs, tool loop, chat)
- `research/rag65/`: training rounds (`build_v4.py`, `build_v5.py`, `paired.py`)
- `research/ai_factory_pilot/`: AI Factory data pipeline (prepare, generate, review, merge)
- `research/leonardo_automation/`: unattended controllers and watchdog on Leonardo
- `research/assistant_package/build_and_smoke.sh`: GGUF build and CPU smoke test
