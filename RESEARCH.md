# DistillKit — State of the Art Research Notes

*Compiled 2026-09-30 for the European AI Hackathon (Oct 6–29, 2026).*

**Scope:** distilling a large open-weight **teacher** into a laptop-sized **student** using **only generated question/answer text**: no logits, no shared tokenizer. The demo is an **HPC Assistant** covering Slurm, CUDA, MPI and profiling.

---

## 0. TL;DR — the recommended recipe

| Stage | Recommendation | Why |
|---|---|---|
| Framing | **Sequence-level KD**: fine-tune the student on teacher-generated text | Works with any tokenizer or model family; the teacher can be a vLLM model or an API |
| Student | **Qwen3.5-2B** (or 0.8B); fall back to **Qwen3-1.7B** if tooling breaks | Newest small Apache-2.0 models (released Mar 2026). Qwen3 is the safer, well-supported fallback |
| Teacher | A 27B–35B-class open model (e.g. **Qwen3.6-27B / Qwen3.8-27B**) run with vLLM, optionally **plus a second teacher** | Stronger models are *not always* better teachers; mixing teachers helps small students |
| Questions | Generated **from document chunks**, diversified by **persona × task type × difficulty** | Avoids mode collapse; ties questions to real docs |
| Answers | **Grounded**: the teacher sees the source chunk; **thinking off**; **4–8 answers per question** | Grounding reduces teacher hallucination; multiple answers are the cheapest way to scale data |
| Filtering | Dedup → **executable checks** (`sbatch --test-only`, `nvcc`, `mpicc`) → LLM judge → decontamination | Verified data is the headline contribution and the main ablation |
| Knowledge | **RAFT-style training + local RAG at inference** | Fine-tuning small models on *new facts* increases hallucination; retrieval supplies the facts |
| Training | Full fine-tune (small enough) or high-rank LoRA; loss on completions only; 2–3 epochs | Correctly configured LoRA matches full fine-tuning |
| Export | GGUF **Q4_K_M + Q8_0 with an imatrix computed on domain text** | The imatrix cuts quantization loss; evaluate every quant level |
| Eval | **Split by source document**; compare the student to its base model, the base model + RAG, and the teacher | Makes the "beats base" claim credible |

---

## 1. What we are actually doing: sequence-level distillation

- **Classic (logit) knowledge distillation** matches the teacher's full token distribution. That needs logits, and in practice the same tokenizer.
- **Sequence-level knowledge distillation** (Kim & Rush, 2016) trains the student on the teacher's *output sequences* with normal cross-entropy. This is what DeepSeek-R1-Distill, OpenThoughts, s1, Alpaca/Vicuna-style models and most "distilled" small models actually do.
- **Consequence:** the framework is teacher-agnostic, so any model that emits text can be a teacher. The research question moves from the loss function to **the data**: how it is generated, how diverse it is, and how it is verified. That is where DistillKit should compete.

> Reviewers may say "this is just SFT on synthetic data." The answer: yes, and in the current literature that *is* the SOTA recipe for small models (see §3). Our contribution is a reusable, verified, domain-pluggable pipeline plus ablations showing which parts matter.

**Where logit-based work is heading, for context only.** Cross-tokenizer **on-policy distillation** now exists: HF's GOLD in TRL, and the Thinking Machines "On-policy distillation" write-up. It needs teacher log-probs of *student* samples at training time. Keep it as a v2 stretch goal; it is out of scope for v1.

---

## 2. Model choice (landscape as of Sep 2026)

**Students (≤ 2B):**

| Model | Notes |
|---|---|
| **Qwen3.5-2B / 0.8B** (Mar 2026, Apache-2.0) | Hybrid Gated-DeltaNet architecture, natively multimodal, 262k context, thinking off by default. The model card warns that 2B is *"prone to thinking loops"*, which is another reason to train it non-thinking. **Verify on day 1** that TRL training and llama.cpp GGUF conversion both work for this newer architecture. |
| **Qwen3-1.7B / 0.6B** (2025) | Mature tooling everywhere (TRL, llama.cpp, Ollama). **The safe fallback.** |
| Gemma 4 e2b | Strong edge model (~1.5 GB at Q4_K_M), Apache-2.0. A good second student to show the framework is model-agnostic. |
| SmolLM3 / SmolLM 1.7B | Fully open data, weaker. Could be a third data point. |

**Base or instruct student?** HPC-Coder-V2 found that fine-tuning **base** 1.3B models beat instruct variants by about 4 percentage points on parallel-code generation. This is worth a cheap ablation, because for a chat assistant the instruct model's formatting habits may still win.

**Teachers:** a strong 27–35B dense or MoE model is the sweet spot for throughput on a few GPUs, e.g. Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B (fast, sparse), or Qwen3.5-122B-A10B if you get multiple nodes.
- **OpenThoughts:** *"models with better performance are not necessarily better teachers."* QwQ-32B was a better teacher than DeepSeek-R1.
- **HPC-Coder-V2:** teacher choice mattered by up to about 6 percentage points (Llama-3-70B data beat DBRX data).
- → **Ablation: compare two teachers**, and consider mixing them.

---

## 3. Synthetic data generation — SOTA practices

### 3.1 Seeds: generate *from documents*, not from thin air

- Chunk the source corpus: Slurm man pages and docs, CUDA Programming Guide and Best Practices Guide, OpenMPI docs, Nsight/`nvprof` docs, university HPC center docs (many are CC-licensed), and forum Q&A.
- **Keep the provenance** (`doc_id`, `chunk_id`) on every generated example. That is needed for the train/eval split (§6) and for RAFT (§4).
- Prior art:
  - **HPC-Coder-V2 / HPC-INSTRUCT:** 125k code seeds from The Stack v2 → 122k samples using 4 templates (write, translate, optimize, parallelize) and 4 teachers.
  - **HPC-LLM (2026):** 9k–24k Q&A pairs generated from university HPC docs, QLoRA on Llama-3.1-8B plus RAG → approaches Qwen2.5-14B.
- **Document rephrasing for knowledge:**
  - **EntiGraph** (entity-graph → synthetic text), **Active Reading** (the model first proposes its own study strategies, then rewrites), **SPA** (a small set of well-designed prompts at scale, a strong baseline), and **Synthetic Mixed Training** (synthetic Q&A + synthetic documents *beat RAG* on QuALITY).
  - These matter if you want knowledge *in the weights*.

### 3.2 Diversity: avoid mode collapse

LLM generators fall into repetitive patterns. The standard countermeasures:

- **Personas** (Persona Hub): "new PhD student porting a code to GPUs", "sysadmin debugging a stuck job", "ML engineer running multi-node DDP", …
- **Task-type taxonomy:** conceptual explanation, write a script, debug an error message, optimize, translate (e.g. PBS→Slurm, OpenMP→CUDA), compare options, interpret profiler output.
- **Difficulty levels** and **question formats** (short, long with pasted logs, multi-turn follow-ups).
- **Magpie-style** prompting (feed only the chat-template prefix so the model writes the user turn) as a complementary source of natural-sounding questions.
- **Sampling:** a combinatorial grid of chunk × persona × task type × difficulty, not a single prompt repeated.

### 3.3 Answers

- **Grounded generation:** give the teacher the source chunk and instruct it to answer from it. This cuts teacher hallucination and fits RAFT (§4).
- **Thinking OFF for the target answer.** Small students learn short, direct answers better. See *"Small Models Struggle to Learn from Strong Reasoners"* (2025): ≤3B students often *degrade* on long chain-of-thought from strong teachers. Their **Mix Distillation** (mixing short and long reasoning, or large- and small-teacher outputs) fixes this. Optional ablation: plain answers vs. a small fraction of brief reasoning.
- **Multiple answers per question.** OpenThoughts: sampling **4×–16× answers per question** is an effective, cheap way to scale data and gave significant gains. It is also exactly the workload that justifies the cluster.
- **Consistent answer style:** a fixed system prompt, runnable code in fenced blocks, and a stated assumption when the cluster configuration is unknown.

### 3.4 Filtering and verification (our headline contribution)

Order matters: **dedup → cheap checks → expensive checks.**

1. **Deduplication:** MinHash near-duplicate removal on questions (threshold ~0.9), then embedding-based semantic dedup. It typically removes 10–30% of a 100k set and *improves* results.
2. **Executable / static checks** (domain-specific, fully automatic):
   - Slurm: `sbatch --test-only` on a real cluster, or a static `#SBATCH` directive linter offline.
   - CUDA: `nvcc -c` compile check (compile only; running the code is optional).
   - MPI: `mpicc`/`mpicxx` compile; optionally run with `mpirun -np 2` on tiny inputs.
   - Shell: `bash -n`, `shellcheck`.
3. **LLM-as-judge** against the source chunk (faithfulness, correctness, helpfulness), using a *different* model family from the teacher to reduce self-preference.
4. **Best-of-N selection:** among the N answers, keep those that pass the checks and score highest (rejection sampling).
5. **Decontamination:** drop training items too similar to eval questions (n-gram + embedding).

> **Caveat for the proposal:** OpenThoughts found that answer verification and filtering gave *no significant gains* for math/code reasoning at their scale. That makes **"filtered vs. unfiltered" a real, open ablation** in our domain, not a foregone conclusion. Report it honestly either way.

---

## 4. The knowledge problem: fine-tuning ≠ reliable fact injection

- **Gekhman et al. (EMNLP 2024), "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?":** the more *new* facts in the fine-tuning data, the more the model hallucinates. Models learn new facts slowly, and doing so makes them less calibrated.
- A 2B student will not memorize every Slurm flag. **HPC-LLM (2026)** got its best results by **combining domain fine-tuning with RAG**.
- **RAFT (Retrieval-Augmented Fine-Tuning):** train on (question + retrieved chunks, *including distractor chunks*) → answer that cites the relevant chunk. The student learns to *use* retrieved docs and ignore noise.

**Recommendation.** Ship the HPC Assistant as a **distilled student + local RAG** over the same docs (a small embedding model plus a local vector store, still under 4 GB total). Train a mix of:
- closed-book examples (general HPC skills, script patterns), and
- RAFT examples (question + 1 gold chunk + 2–4 distractors → grounded answer).

This turns a weakness into an ablation: *student alone vs. student+RAG vs. RAFT-student+RAG*.

---

## 5. Training the student

- **Full fine-tuning vs. LoRA:**
  - A 2B model fully fine-tunes comfortably on one node.
  - **"LoRA Without Regret" (Thinking Machines, 2025):** correctly configured LoRA (applied to *all* layers including MLP, high rank, **~10× the full-fine-tuning learning rate**) matches full fine-tuning in SFT at about 2/3 of the compute. Low-rank LoRA falls off once the adapter runs out of capacity on large datasets.
  - → Default to **full fine-tuning** for a 2B student with 50k+ examples; offer LoRA as a config option.
- **Loss masking:** compute loss only on the completion (the TRL default for conversational data). HPC-Coder-V2 found little difference in their setting, but it is the standard.
- **Data scale:** HPC-Coder-V2 saw a 1.3B model's MPI performance **plateau around 6k examples** of that subtype. Expect diminishing returns per topic → scale **diversity**, not raw count. Plot a **data-scaling curve** (10k / 25k / 50k / 100k); it is a cheap, compelling result.
- **Starting hyperparameters (full fine-tuning):** LR 1e-5 to 2e-5, cosine schedule with warmup of about 3%, 2–3 epochs, sequence packing, bf16, effective batch around 64–128 sequences.
- **Distributed:** TRL `SFTTrainer` + `accelerate` with FSDP or DeepSpeed ZeRO-2/3. It runs data-parallel across GPUs; a 2B model doesn't need model parallelism.

### Stretch goals beyond SFT (still no logits)

- **DPO with verifier-derived pairs:** chosen = a teacher answer that passed the checks; rejected = a student answer that failed the checks, or a failed teacher sample. Uses the check signals we already compute.
- **Rejection-sampling self-improvement:** sample from the student, keep answers that pass the checks and the judge, and retrain.
- **On-policy distillation (v2):** GOLD or the Thinking Machines recipe. Needs teacher log-probs of student samples, but works across tokenizers.

---

## 6. Evaluation design

1. **Split by source document before generating anything.** Hold out ~10–15% of documents (and some whole topics) for eval only. This prevents the leakage that would otherwise make "beats base" meaningless.
2. **Eval set:** about 150–300 questions from held-out docs, **human-reviewed**, tagged by topic (Slurm / CUDA / MPI / profiling) and by check type (`slurm | cuda | mpi | none`).
3. **Metrics:**
   - **Pass rate on executable checks** (the hard metric).
   - **LLM-judge score** against the reference with a fixed rubric (pairwise vs. base is more stable than absolute scores).
   - **Hallucination rate:** judge whether the answer invents flags or APIs. Option: auto-check that every `#SBATCH --flag` exists in the Slurm man page.
   - **Latency and memory:** tokens/s and peak RAM on a laptop CPU for each quant level.
4. **Baselines:** the student's base model (zero-shot), base + RAG, distilled student, distilled student + RAG, and the teacher (upper bound).
5. **External benchmarks, for credibility:**
   - **ParEval** (420 parallel-code tasks across OpenMP/MPI/CUDA, etc.) for the code-generation slice.
   - Compare against **HPC-LLM** / **ChatHPC** numbers where comparable.

---

## 7. Quantization and export

- Convert the HF model with llama.cpp `convert_hf_to_gguf.py` → `llama-quantize`.
- **Importance matrix (imatrix):** compute it with `llama-imatrix` on *domain* text (a sample of our training answers). It typically reduces quantization perplexity loss meaningfully; this matters most at Q4 and below and for small models.
- Export **Q8_0** (near-lossless) and **Q4_K_M** (default). Optionally add Q3_K_M/IQ4_XS to find the "breaking point".
- **Re-run the full eval at every quant level** → a "quality vs. size" curve.
- Size check: a 2B model at Q4_K_M is about 1.2–1.5 GB. The "<4 GB RAM" target is easy, so aim to show **a phone-class/CPU tokens-per-second result** as well.
- Ship an **Ollama `Modelfile`** with the correct chat template and system prompt, so users get one-command install.

---

## 8. Tooling (don't reinvent)

| Need | Options |
|---|---|
| Teacher inference | **vLLM** (offline batch `LLM.generate` or an OpenAI-compatible server), tensor parallelism across GPUs |
| SDG orchestration | **distilabel** (HF; DAG of steps, vLLM backend), **NeMo Data Designer** (NVIDIA; declarative configs, fits an NVIDIA-sponsored event well), Bespoke **Curator** |
| Dedup | `datasketch` (MinHash), `text-dedup`, embedding + FAISS |
| Training | **TRL `SFTTrainer`/`DPOTrainer`** + accelerate (FSDP/DeepSpeed); **Unsloth** for single-GPU LoRA |
| Eval | custom harness + `lm-eval-harness` for general-capability regression checks |
| Export | llama.cpp (GGUF, imatrix), Ollama |

**Build decision:** keep DistillKit's own code thin. Use plain vLLM for the core path (fewer moving parts on an HPC cluster with Slurm and containers), and structure the modules so a distilabel or Data Designer backend can be swapped in.

---

## 9. Ablations mapped to cluster usage (for the hackathon)

| # | Ablation | Compute |
|---|---|---|
| A1 | Filtered (checks + judge) vs. unfiltered data | Teacher generation at N×, checks on CPU nodes |
| A2 | 1 vs. 4 vs. 8 answers per question | Teacher generation |
| A3 | Teacher A vs. B vs. mixed | Teacher generation ×2 |
| A4 | Closed-book SFT vs. RAFT (+ RAG at inference) | Training ×2 |
| A5 | Data scale 10k → 100k | Training ×4 (small) |
| A6 | Student 0.8B vs. 2B (and base vs. instruct) | Training |
| A7 | Q8_0 vs. Q4_K_M (± imatrix) | Eval only |

A1–A3 are what justify leadership-class GPUs: **large-scale multi-sample teacher generation**. Student training is comparatively cheap.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Qwen3.5 hybrid architecture not yet supported in some tool (TRL/FSDP/llama.cpp GGUF) | Test end-to-end on day 1 with 100 examples; fall back to Qwen3-1.7B |
| Eval leakage | Split by document before generation, plus n-gram/embedding decontamination |
| Teacher hallucinations baked into training data | Grounded generation, checks, a judge from a different model family |
| Student hallucinating flags | RAG + RAFT; flag-existence checker in the eval |
| `sbatch --test-only` needs a Slurm controller | Run on the hackathon cluster; static linter fallback for offline and CI use |
| Generation is slower than planned | Start with a 5k pilot, measure throughput, then scale |
| Licensing of seed docs | Record the license per source; publish generated data only from permissively licensed sources |

---

## 11. Suggested pre-event prep (before Oct 6)

1. Collect and chunk the seed corpus with provenance; decide the held-out document split.
2. Write ~50 eval questions by hand as the gold core of the eval set.
3. Build the pipeline at toy scale locally (tiny teacher/student or an API teacher): generate → filter → SFT → GGUF → eval.
4. Get the check scripts (Slurm linter, `nvcc` compile wrapper) working in a container.
5. Prepare Slurm job scripts for vLLM generation and multi-GPU training.

---

## References

- Kim & Rush (2016). *Sequence-Level Knowledge Distillation.* EMNLP.
- Guha et al. (2025). *OpenThoughts: Data Recipes for Reasoning Models.* [arXiv:2506.04178](https://arxiv.org/abs/2506.04178)
- Li et al. (2025). *Small Models Struggle to Learn from Strong Reasoners.* [ACL Findings](https://aclanthology.org/2025.findings-acl.1301/) · [project page](https://small-model-gap.github.io/)
- Gekhman et al. (2024). *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* EMNLP. arXiv:2405.05904
- Zhang et al. (2024). *RAFT: Adapting Language Model to Domain Specific RAG.*
- Chaturvedi, Nichols, Singh & Bhatele (2024). *HPC-Coder-V2: Studying Code LLMs Across Low-Resource Parallel Languages.* [arXiv:2412.15178](https://arxiv.org/abs/2412.15178)
- Shahin & Alsmadi (2026). *HPC-LLM: Practical Domain Adaptation and RAG for HPC Support.* [arXiv:2605.16347](https://arxiv.org/abs/2605.16347)
- Nichols et al. *ParEval: Can Large Language Models Write Parallel Code?*
- *LM4HPC: Towards Effective Language Model Application in HPC.* [arXiv:2306.14979](https://arxiv.org/pdf/2306.14979)
- Yang et al. *Synthetic Continued Pretraining (EntiGraph).* · *SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection* [arXiv:2603.22213](https://arxiv.org/pdf/2603.22213) · *Synthetic Mixed Training* [arXiv:2603.23562](https://arxiv.org/html/2603.23562v1)
- (2024). *Scaling Synthetic Data Creation with 1,000,000,000 Personas (Persona Hub).*
- Xu et al. (2024). *Magpie: Alignment Data Synthesis from Scratch.*
- Thinking Machines Lab (2025). [*LoRA Without Regret*](https://thinkingmachines.ai/blog/lora/) · *On-Policy Distillation*
- Hugging Face (2025). [*Unlocking On-Policy Distillation for Any Model Family (GOLD)*](https://huggingfaceh4-on-policy-distillation.hf.space/)
- Lambert et al. (2024). *Tülu 3* (decontamination and data-mixing practice). [arXiv:2411.15124](https://arxiv.org/pdf/2411.15124)
- [NeMo Data Designer](https://arxiv.org/html/2609.17699v1) · [distilabel](https://huggingface.co/docs/hub/en/datasets-distilabel)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B) · [Qwen releases](https://github.com/QwenLM/Qwen3.8) · [llama.cpp quantize / imatrix](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

> **Note:** some items above were verified only from abstracts or summaries during this search pass (notably HPC-LLM and the 2026 synthetic-knowledge papers). Read the full papers before quoting their numbers in the final report.
