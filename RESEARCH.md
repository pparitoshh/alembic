# DistillKit — State of the Art Research Notes

*First compiled 2026-09-30; **revised 2026-10-04** for the finalized design in [GOAL.md](GOAL.md) (Qwen3-32B teacher, Qwen3-4B-Instruct-2507 QDoRA student, FSDP, tool calling, saved top-20 logprobs, one shared Leonardo node).*
*European AI Hackathon, Oct 6–29, 2026.*

**Scope:** distilling a large open-weight **teacher** into a laptop-sized **student**. v1 trains on **generated text only** (sequence-level KD), but the teacher's **top-20 logprobs are saved** so logit-level KD can be added in v2. The demo is an **HPC Assistant** covering Slurm, CUDA, MPI and profiling, **with tool calling**.

---

## 0. TL;DR — the recipe

| Stage | Choice | Why |
|---|---|---|
| Framing | **Sequence-level KD** in v1; top-20 logprobs saved for v2 logit KD | Tokenizer-independent in general; teacher and student here share the Qwen3 tokenizer, so logit KD stays possible later |
| Teacher | **Qwen3-32B**, 4-bit, non-thinking, on **one A100 64 GB** | Fits one GPU on a shared node; Apache 2.0 |
| Teacher serving | **Open decision → recommend vLLM + official `Qwen3-32B-AWQ`**; keep llama.cpp + GGUF Q4_K_M as fallback (§2.2) | vLLM's GGUF path is "highly experimental"; vLLM AWQ has continuous batching and native top-k logprobs with token IDs |
| Student | **Qwen3-4B-Instruct-2507** (decided Oct 4, §2.1) | 2507 update: BFCL-v3 57.6 → 61.9, TAU1-Retail 24.3 → 48.7 |
| Training | **QDoRA** (bnb 4-bit + `use_dora=True`), TRL `SFTTrainer`, `assistant_only_loss`, **FSDP** | DoRA is closer to full FT at low rank; PEFT reports issues with QDoRA under DeepSpeed ZeRO-2 |
| Data | Document-grounded; **~70% prose / ~30% tool-call traces**; persona × task × difficulty grid; gold few-shot anchors | Coverage over volume; consistent format |
| Verification | Dedup → exec checks (`bash -n`, Slurm linter, `nvcc`, `mpicc`) → **tool-call checks** (schema, args, execution) → judge → decontamination | APIGen-style 3-stage check for tool calls |
| Knowledge | Distilled student + **local RAG**; RAFT as stretch | Fine-tuning on new facts raises hallucination |
| Export | GGUF **Q8_0 + Q4_K_M with domain imatrix**, Ollama Modelfile **with tool-call template** | Re-evaluate at every quant level |
| Eval | Split by document; 3 seeds; prose judge + exec checks + **tool-call validity + when-to-call** | Makes "beats base" credible |

---

## 1. Sequence-level KD now, logit-level KD later

- **Classic (logit) KD** matches the teacher's token distribution. It needs teacher logits on the *same vocabulary* as the student.
- **Sequence-level KD** (Kim & Rush, 2016) trains on teacher output text with normal cross-entropy. This is what DeepSeek-R1-Distill, OpenThoughts, s1 and most "distilled" small models do. Our framework stays teacher-agnostic: any model that emits text can be a teacher.
- **Why save top-20 logprobs now:** Qwen3-32B and Qwen3-4B-Instruct-2507 share the Qwen3 tokenizer, so cached teacher logprobs can be used for a KL + CE loss in v2 without re-running the teacher (G13).
- **Caveat on top-k caching:** *Sparse Logit Sampling* (Anshumann et al., ACL 2025) shows that caching top-K probabilities gives a **biased** estimate of the teacher distribution, which hurts performance and calibration. Their **Random Sampling KD** (importance-sampled tokens) is unbiased and needs even fewer stored logits, at < 10% overhead over CE training (tested at 300M–3B). → For v2, either renormalize the top-20 and accept the bias, or store a random-sampled set alongside the top-20. Decide before the big generation run, since it changes what we store.
- **Logprobs from a 4-bit teacher** are the quantized model's distribution, not bf16 Qwen3-32B's. Fine for v2 KD, but state it in the dataset card.
- **Prior art to cite (and a name clash):** Arcee AI already publishes an open-source logit-distillation toolkit called **DistillKit** ([arcee-ai/DistillKit](https://github.com/arcee-ai/DistillKit)), with offline distillation from compressed cached logits. No `distillkit` package showed up on PyPI in our search, but **the name is taken on GitHub**. Consider renaming before the public release (G4), and position against it: ours is sequence-level + verified data + tool calling + laptop export; theirs is logit-level.
- On-policy cross-tokenizer distillation (HF GOLD in TRL; Thinking Machines' write-up) needs teacher logprobs of *student* samples at training time. Out of scope for v1.

---

## 2. Models

### 2.1 Student: Qwen3-4B

| Variant | Notes |
|---|---|
| **Qwen3-4B** (Apr 2025, hybrid) | Thinking on/off via `enable_thinking`. BFCL-v3 57.6 (non-thinking). |
| **Qwen3-4B-Instruct-2507** | Non-thinking only (never emits `<think>`), 262k context, Apache 2.0. **BFCL-v3 61.9, TAU1-Retail 48.7, TAU1-Airline 32.0**, far ahead of the original on agentic tasks. Recommended sampling: T 0.7, top-p 0.8, top-k 20. |

**Decision (Oct 4):** the student is **Qwen3-4B-Instruct-2507**. It has the same architecture and tokenizer, so all GOAL.md reasoning holds, and it is trained for tool use without thinking, which is the mode we want. **The baseline is the same checkpoint, untrained** (G2).

- **Tool-call format (checked against the real templates):** Hermes style. Tools go into the system prompt inside `<tools>` as `{"type": "function", "function": {...}}`; the model writes `<tool_call>{"name": …, "arguments": {…}}</tool_call>`; results come back in a **user** turn inside `<tool_response>`. vLLM (`--tool-call-parser hermes`), llama-server (`--jinja`) and Ollama (with the Qwen3 template in the Modelfile) all parse it into OpenAI `tool_calls`. No Claude Code or other external agent is needed at runtime.
- The hybrid Qwen3-4B adds empty `<think></think>` blocks to some assistant turns, and TRL's training template places them slightly differently from the official template. 2507 has no think blocks, which removes that mismatch.

- **Keep thinking off.** *The Reasoning Trap* (2025) reports that strengthening reasoning **amplifies tool hallucination**; *Small Models Struggle to Learn from Strong Reasoners* (ACL Findings 2025) found ≤3B students degrade on long CoT. Short reasoning before a tool call is fine; long `<think>` blocks are not.
- **Size check:** Qwen3-4B at Q4_K_M is about **2.4–2.5 GB** (perplexity +0.30 vs. full precision, as reported by quant uploaders). That leaves ~1.5 GB for KV cache, the embedding model and the RAG index under the 4 GB target. Keep the default context small (4–8k) in the Modelfile.
- **Laptop speed risk:** CPU decode is memory-bandwidth bound. Desktop CPUs reach ~30–35 tok/s on a 7B Q4_K_M; a 4B is ~1.7× smaller, but laptop DDR4/DDR5 bandwidth is much lower. **≥ 20 tok/s on a typical laptop CPU is not guaranteed.** Measure on the demo laptop in week 1 and report Q4_K_M vs. a smaller quant (or the 1.7B student) if needed.

### 2.2 Teacher: Qwen3-32B, and how to serve it

Qwen3-32B has no 2507 refresh; use it in **non-thinking mode** (`enable_thinking=False`) for answers. There are official Qwen GGUF and AWQ builds.

| Option | Logprobs | Throughput | Risk |
|---|---|---|---|
| **llama.cpp `llama-server` + GGUF Q4_K_M** (~20 GB) | OpenAI-compatible `logprobs` / `top_logprobs` (`n_probs`). Logprobs are taken **before** grammar masking. | Parallel slots, but lower aggregate throughput than vLLM under concurrency | Simple build on Leonardo; check that the response includes **token IDs**, not just strings |
| **vLLM + GGUF** | Native | Unknown | vLLM docs: GGUF support is **"highly experimental and under-optimized"**, may conflict with other features; now needs a plugin. **Avoid.** |
| **vLLM + `Qwen/Qwen3-32B-AWQ`** (~19 GB) | Native; `--max-logprobs` defaults to **20**; `return_tokens_as_token_ids` gives IDs directly | PagedAttention + continuous batching; AWQ runs via the Marlin kernel on Ampere | Needs vLLM on Leonardo (container or uv wheel); A100 has **no FP8**, so FP8 builds are out |

**Recommendation:** close the open decision with **vLLM + Qwen3-32B-AWQ** on one A100 64 GB. It has the same 4-bit footprint, the best throughput, and IDs + top-20 logprobs in one call. Keep the llama.cpp/GGUF path as the fallback if vLLM can't be installed. Both expose an OpenAI-compatible API, so our `teacher.py` client doesn't change.

- **Day-1 smoke test:** request 1 completion with `logprobs=True, top_logprobs=20`, check that the IDs decode back to the text with the **student's** tokenizer, and log tokens/s at concurrency 1, 16 and 64.
- **Storage estimate:** 20 × (token ID + float) per generated token. At ~10k examples × ~400 answer tokens × 20 entries, that is ~80M entries: **hundreds of MB as fp16 + int32 in Parquet/NPZ**, several GB as JSON. Store binary, not JSON.

---

## 3. Synthetic data generation

### 3.1 Seeds from documents

- Chunk the corpus with provenance (`doc_id`, `chunk_id`): Slurm man pages and docs, CUDA Programming/Best Practices guides, OpenMPI docs, Nsight docs, university HPC center docs (many CC-licensed), Leonardo user guide. Record the **license per source** (G10 publishes only permissive sources).
- Prior art: **HPC-Coder-V2** (122k samples, 4 templates × 4 teachers; 1.3B MPI performance plateaued around **6k examples per subtype**) and **HPC-LLM** (2026; 9k–24k Q&A from university HPC docs, QLoRA Llama-3.1-8B + RAG, approaching Qwen2.5-14B).

### 3.2 Diversity

- **Grid sampling:** chunk × persona (new PhD student, sysadmin, ML engineer doing multi-node DDP, …) × task type (explain, write a script, debug an error, optimize, translate PBS→Slurm, interpret profiler output, *call a tool*) × difficulty.
- **Magpie-style** prompting as an extra source of natural questions.
- **Gold examples as anchors:** GOAL.md §6 asks for one hand-written prose answer and one prose + tool-call trace. Use them as few-shot examples in every teacher prompt to fix the format; rotate phrasings so the student doesn't copy the anchor's wording.

### 3.3 Prose answers

- **Grounded:** the teacher sees the chunk and answers from it; non-thinking; short and direct.
- **Multiple answers per question** (OpenThoughts: 4×–16× is a cheap, effective way to scale). With one GPU, budget 1–2 answers/question for the 10k pilot and spend extra samples on failure areas (GOAL.md §6 step 5).

### 3.4 Tool-calling traces (new)

- **Format:** Qwen3 uses **Hermes-style** tool calls: tools as JSON schemas inside `<tools>` in the system prompt; the model emits `<tool_call>{"name": …, "arguments": …}</tool_call>`; results come back as a `tool` role message. vLLM parses this with `--tool-call-parser hermes`. Generate traces **in this exact format** so that SFT, vLLM, llama.cpp and Ollama agree.
- **Trace shape** (GOAL.md §6): question → short reasoning → tool call → **tool result** → final answer. The tool result has to come from somewhere: implement the 5–10 tools as **deterministic mock functions** (fake `squeue`/`sacct`/`sinfo`/log files/`nvidia-smi` output) so results are realistic, reproducible, and checkable.
- **Mix in "don't call" cases.** *When2Call* (NAACL 2025) evaluates *when* to call, when to ask a follow-up, and when to say the tools can't help, and found preference training beat plain SFT on this. *SimpleToolHalluBench* targets abstention when no suitable tool exists. → Include questions answerable directly, questions with missing required arguments (ask back), and questions no tool covers. Something like 30% tool traces split roughly 2:1 between "call" and "don't call / ask" is a reasonable starting point; tune it from the eval.
- **Training format:** TRL's `SFTTrainer` accepts a `tools` column (JSON schemas) that the chat template renders into the system prompt. With `assistant_only_loss=True`, TRL patches the Qwen3 template with `{% generation %}` markers so loss falls only on assistant turns, i.e. **not on tool results**. Check the patched template on one sample before the first real run.

### 3.5 Verification

Order: **dedup → cheap checks → expensive checks.**

1. **Dedup:** MinHash on questions (~0.9), then embedding dedup.
2. **Prose executable/static checks:** Slurm `#SBATCH` linter vs. man-page flags + `bash -n`/`shellcheck` (offline), `sbatch --test-only` on Leonardo; `nvcc -c`; `mpicc`.
3. **Tool-call checks**, following **APIGen**'s three stages (NeurIPS 2024; 7B models trained on its verified data reached SOTA on BFCL):
   - *format*: valid JSON, a known tool name, required args present, types match the schema;
   - *execution*: run the call against the mock tool, no errors;
   - *semantic*: a judge checks that the call and final answer actually answer the question.
4. **LLM judge** against the chunk, from a different model family than the teacher.
5. **Decontamination** against the eval set (n-gram + embedding).

> Open question (G5): OpenThoughts found little gain from answer filtering in math/code reasoning. Whether it helps here is a real ablation; report it either way.

---

## 4. Knowledge: fine-tuning ≠ fact injection

- **Gekhman et al. (EMNLP 2024):** the more *new* facts in the fine-tuning data, the more the model hallucinates.
- **HPC-LLM (2026)** got its best results combining domain fine-tuning with RAG.
- **RAFT:** train on question + gold chunk + distractors → grounded answer (G8 stretch).
- **Ship student + local RAG** over the same docs (small embedding model + local vector store, inside the 4 GB budget). Tool calls cover live state (jobs, GPUs, logs); RAG covers documentation; the weights cover style and HPC skills.

---

## 5. Training the student: QDoRA + FSDP

- **DoRA** splits each weight into magnitude and direction: LoRA's A/B update the direction, a trainable vector updates the magnitude. Enabled with `LoraConfig(use_dora=True)`; PEFT supports it on bitsandbytes-quantized linear layers ("**QDoRA**"). Answer.AI showed FSDP + QDoRA on Llama 3 matching or beating full fine-tuning at a fraction of the memory.
- **Cost:** DoRA is roughly **1.5–1.8× slower** than LoRA in training (more with dropout > 0; PEFT's DoRA path is faster with `lora_dropout=0`). Budget for it in the seed × ablation plan. Memory overhead is small without caching.
- **FSDP, not ZeRO-2:** PEFT docs report issues with QDoRA under DeepSpeed ZeRO-2. FSDP + 4-bit needs **`bnb_4bit_quant_storage` set to the training dtype (bf16)** and recent bitsandbytes/accelerate/transformers/TRL (PEFT ≥ 0.10 enabled QLoRA under FSDP/ZeRO-3).
- **A 4B QDoRA run fits on one A100.** FSDP is for speed when 2–4 GPUs are free, not for memory. On a shared node, the default should be **one seed per GPU in parallel**; use FSDP only for single long runs. That also gives us seed variance cheaply.
- **Hyperparameters (starting point, re-tune on day 1):** r 16/α 32 on all linear layers, lr ~1e-4, 2–3 epochs, cosine with ~3% warmup, completion-only loss, packing, bf16 compute. One short lr sweep, then 3 seeds at the chosen point. *LoRA Without Regret* (Thinking Machines, 2025): LoRA on all layers at ~10× the full-FT lr matches full FT; low rank falls behind once data outgrows adapter capacity, so try r 64 if the 10k run underfits.
- **Data scale:** expect plateaus per subtype (~6k in HPC-Coder-V2). Scale *coverage* (every tool, every topic) before raw count.

### Stretch beyond SFT

- **DPO (G9)** with verifier-derived pairs: chosen = a passing teacher answer, rejected = a failing student answer. When2Call suggests preference training is especially useful for the when-to-call decision.
- **Logit KD (G13):** KL(teacher top-k ‖ student) + CE using the saved logprobs (see §1 on top-k bias). TRL's GKD trainer covers the online variant.
- **Curriculum (G14):** *Pedagogically-Inspired Data Synthesis for LM Knowledge Distillation* (ICLR 2026, "IOA": Identifier → Organizer → Adapter) finds student gaps, orders knowledge by prerequisites with gradual difficulty, and adapts representations to the student. It reports students keeping 94.7% of teacher performance on DollyEval (LLaMA-3.x / Qwen2.5 students). Our "generate more of what the student fails on" loop is a light version of its Identifier step.

---

## 6. Evaluation design

1. **Split by source document before any generation**; hold out ~10–15% of docs and some whole topics.
2. **Eval set (~150–200 Qs, human-reviewed)**, tagged by topic and check type (`slurm | cuda | mpi | tool | none`):
   - **Prose:** pairwise LLM judge vs. base, both orders, judge sees the reference chunk.
   - **Executable:** sbatch scripts parse/lint; CUDA/MPI snippets compile.
   - **Tool calling:** BFCL-style **AST match** of the call (name + arguments vs. expected) instead of string match, plus **relevance/irrelevance** items (should call / should not call / should ask), plus executing the call against the mock tools.
3. **Hallucinated flags:** share of `#SBATCH --flag`/`srun --flag` not in the man pages (implemented in `checks.py`).
4. **Baselines:** base student (same checkpoint), base + RAG, distilled student, distilled + RAG, teacher (upper bound for "≥ 80% of teacher").
5. **Seeds:** ≥ 3 training seeds per setting, mean ± spread; on a small eval set, seed noise can be several points. A 150–200 question set tightens this; report a bootstrap CI as well.
6. **Speed and memory:** tokens/s and peak RAM on the laptop CPU per quant level.
7. **External sanity checks:** a BFCL subset (general tool-calling regression: did HPC tuning break general tool use?), ParEval for parallel code.

---

## 7. Quantization and export

- llama.cpp `convert_hf_to_gguf.py` → `llama-imatrix` on **domain text** (training answers + tool traces) → `llama-quantize` to **Q8_0** and **Q4_K_M** (add Q5_K_M / Q3_K_M / IQ4_XS for the ≥ 3-level Pareto in G6).
- **Merge DoRA into the bf16 base before conversion.** Don't convert from the 4-bit base: load the base in bf16, merge the adapter, then convert.
- **Ollama Modelfile must carry the tool-calling template** (Qwen3's `<tools>` / `<tool_call>` format) or Ollama tool calls won't work. Test `ollama run` with a tool request in the demo.
- Re-run the full eval at every quant level (target: Q4_K_M within 2 pp of bf16).

---

## 8. Tooling

| Need | Choice |
|---|---|
| Teacher serving | **vLLM** (AWQ, `--max-logprobs 20`, `--tool-call-parser hermes`); llama.cpp `llama-server` fallback |
| Generation client | Our OpenAI-compatible client (`teacher.py`) with tenacity retries; chunked output files for checkpointing |
| Dedup | `datasketch` (MinHash), embedding + FAISS |
| Training | TRL `SFTTrainer` (`tools` column, `assistant_only_loss`), PEFT DoRA, bitsandbytes 4-bit, accelerate FSDP |
| Tool checks | JSON Schema validation + mock tool implementations |
| Export | llama.cpp (GGUF, imatrix), Ollama |
| Optional orchestration | distilabel / NeMo Data Designer / Curator; keep our code thin and swappable |

---

## 9. Ablations mapped to the shared node

| # | Ablation | Compute |
|---|---|---|
| A1 | Filtered vs. unfiltered (G5) | Generation once, train ×2 × 3 seeds |
| A2 | Data scale (e.g. 2.5k / 5k / 10k) | Train only |
| A3 | Tool-trace share and "don't call" share | Train only |
| A4 | Quant levels Q8_0 / Q5_K_M / Q4_K_M / Q3_K_M (± imatrix) | Eval only |
| A5 | Student size 1.7B vs. 4B (G6, stretch G11) | Train |
| A6 | Closed-book vs. RAFT (+ RAG) (G8) | Train |

Generation is the GPU-hour-heavy step; on one shared GPU it is the **critical path**. Train runs are cheap and parallelize across free GPUs.

---

## 10. Leonardo facts (from CINECA docs)

- **Booster node:** 1× 32-core Xeon 8358, 512 GB RAM, **4× A100 64 GB** (NVLink 3.0).
- **QOS wall times:** `boost_usr_prod` **24 h**, `boost_qos_dbg` 30 min (good for smoke tests), `boost_qos_lprod` 4 days. Our hackathon node may have its own reservation; confirm.
- **Not stated in the docs we read:** internet on compute nodes, quotas for `$WORK`/`$SCRATCH`, container runtime. Ask the mentors; assume **no internet** and pre-download weights from login nodes.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Teacher can't return top-20 logprobs with token IDs in our stack | Day-1 smoke test (§2.2); vLLM AWQ primary, llama.cpp fallback; worst case, skip logprobs (v2 only, not a must-have) |
| One shared GPU makes generation too slow for 10k | Measure tokens/s on day 1; chunked jobs that resume; 1 answer/question; shorter answers; run off-peak |
| QDoRA + FSDP breaks on the cluster's library versions | Default to single-GPU runs (one seed per GPU); FSDP only when proven; plain QLoRA as fallback (one flag) |
| Tool traces with inconsistent format | Gold anchors, Hermes format end to end, format checker, test SFT template rendering |
| Student over-calls tools or invents tools/args | "Don't call / ask back" examples, irrelevance eval items, schema validation; DPO as stretch |
| Laptop < 20 tok/s | Measure early; smaller quant or 1.7B student; report the trade-off on the Pareto plot |
| Name clash with Arcee's DistillKit | Decide on the release name before G4 |
| Eval leakage | Doc-level split before generation + decontamination |
| Licensing of seed docs | Record license per source; publish data only from permissive sources |

---

## References

- Kim & Rush (2016). *Sequence-Level Knowledge Distillation.* EMNLP.
- Anshumann et al. (2025). *Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs.* ACL 2025. [arXiv:2503.16870](https://arxiv.org/abs/2503.16870)
- Arcee AI. *DistillKit* (logit distillation toolkit). [github.com/arcee-ai/DistillKit](https://github.com/arcee-ai/DistillKit)
- Liu et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* [github.com/NVlabs/DoRA](https://github.com/nvlabs/DoRA) · [PEFT DoRA docs](https://huggingface.co/docs/peft/en/package_reference/lora_variant_dora) · [Answer.AI: FSDP QDoRA for Llama 3](https://www.answer.ai/posts/2024-04-26-fsdp-qdora-llama3.html)
- [PEFT v0.10.0: QLoRA with DeepSpeed ZeRO-3 and FSDP](https://newreleases.io/project/pypi/peft/release/0.10.0)
- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [vLLM GGUF docs](https://docs.vllm.ai/en/stable/features/quantization/gguf/) · [vLLM engine args (`--max-logprobs`)](https://docs.vllm.ai/en/stable/configuration/engine_args.html)
- [llama.cpp server: OpenAI-compatible logprobs (#10783)](https://cdn04132025.gitlink.org.cn/replica/llama.cpp/commit/57bb2c40cd94c5a09f5210ed8264cc93b21c4b7e)
- [TRL SFTTrainer docs (`tools`, `assistant_only_loss`)](https://huggingface.co/docs/trl/main/en/sft_trainer)
- Liu et al. (2024). *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets.* NeurIPS 2024. [arXiv:2406.18518](https://huggingface.co/papers/2406.18518)
- (2025). *When2Call: When (not) to Call Tools.* NAACL 2025. [arXiv:2504.18851](https://www.arxiv.org/abs/2504.18851)
- (2025). *The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination.* [arXiv:2510.22977](https://arxiv.org/pdf/2510.22977)
- He et al. (2026). *Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation.* ICLR 2026. [arXiv:2602.12172](https://arxiv.org/abs/2602.12172)
- Guha et al. (2025). *OpenThoughts: Data Recipes for Reasoning Models.* [arXiv:2506.04178](https://arxiv.org/abs/2506.04178)
- Li et al. (2025). *Small Models Struggle to Learn from Strong Reasoners.* [ACL Findings](https://aclanthology.org/2025.findings-acl.1301/)
- Gekhman et al. (2024). *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* EMNLP. arXiv:2405.05904
- Zhang et al. (2024). *RAFT: Adapting Language Model to Domain Specific RAG.*
- Chaturvedi et al. (2024). *HPC-Coder-V2.* [arXiv:2412.15178](https://arxiv.org/abs/2412.15178)
- Shahin & Alsmadi (2026). *HPC-LLM: Practical Domain Adaptation and RAG for HPC Support.* [arXiv:2605.16347](https://arxiv.org/abs/2605.16347)
- Thinking Machines Lab (2025). [*LoRA Without Regret*](https://thinkingmachines.ai/blog/lora/) · *On-Policy Distillation*
- Persona Hub (2024); Magpie (2024); Tülu 3 (decontamination) [arXiv:2411.15124](https://arxiv.org/pdf/2411.15124)
- [CINECA Leonardo user guide](https://docs.hpc.cineca.it/hpc/leonardo.html) · [llama.cpp quantize / imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

> **Verification note:** several items were checked only from abstracts, docs snippets or search summaries (Sparse Logit Sampling, When2Call, The Reasoning Trap, IOA/ICLR 2026, HPC-LLM, DoRA speed overhead, Qwen3-4B Q4_K_M size). The llama.cpp token-ID behaviour and the logprob storage size are our own estimates; confirm both in the day-1 smoke test. Read the full papers before quoting numbers in the final report.
