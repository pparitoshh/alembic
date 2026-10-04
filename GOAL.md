# DistillKit — Goals & Expected Outcomes

*European AI Hackathon · Oct 6–29, 2026 · Team of 5*
*Companion to [RESEARCH.md](RESEARCH.md), which covers the state of the art and the design rationale.*
*Updated Oct 4, 2026: teacher/student finalized, QDoRA + FSDP, tool calling, shared single node, eval-first ordering. Student set to Qwen3-4B-Instruct-2507.*

---

## 1. Mission

**Make large open-weight models practical to run outside the data centre.**

We use the cluster once, to turn a large teacher model into a small, specialised student. The student then runs forever on a laptop: private, offline, and free.

## 2. The problem

- **For developers:** existing distillation scripts are tied to specific architectures and often assume a shared tokenizer. There is no clean, reusable package that takes *any* teacher and *any* downstream task and produces a laptop-sized model plus honest evidence of how good it is.
- **For HPC users:** people on clusters constantly ask the same Slurm, CUDA, MPI and profiling questions. General chatbots hallucinate flags and APIs, cloud assistants are often not allowed on sensitive clusters, and there is no private, offline assistant for cluster work.

## 3. What we are building

An **MIT-licensed, tokenizer-independent, sequence-level distillation framework** (the student learns from teacher-generated Q&A text; loss on answer tokens only). Teacher **top-20 logprobs are saved** during generation so logit-level KD can be added later without re-running the teacher. Four pluggable modules:

1. **Teacher** — any model that emits text (Hugging Face model, GGUF, or an OpenAI-compatible endpoint).
2. **Data generation** — document-grounded, diversified question generation; prose answers and **tool-calling traces**; **verification** (dedup, executable checks, tool-call validation, LLM judge, decontamination).
3. **Student training** — TRL SFT with **QDoRA** (4-bit base + DoRA adapters), **FSDP** for multi-GPU.
4. **Quantize, export and evaluate** — GGUF (Q8_0 / Q4_K_M with imatrix), an Ollama package, and an evaluation harness run at every size and quantization level.

**Demo use case: an HPC Assistant**, a 4B student specialised in Slurm, CUDA, MPI and profiling, with **tool calling** (e.g. job status, job submission, log reading, GPU availability), shipped with a small local RAG index over the same documentation.

### Final model choices

| Role | Model | Format | Why |
|---|---|---|---|
| **Teacher** | Qwen3-32B (official Qwen) | GGUF Q4_K_M (~20 GB) | Fits on one A100 64 GB on a shared node |
| **Student** | Qwen3-4B-Instruct-2507 | bitsandbytes 4-bit + DoRA (QDoRA) | Capacity for domain + tool calling; non-thinking only; Q4_K_M export ~2.5 GB |

- **Why the 2507 Instruct variant:** the Jul 2025 update of Qwen3-4B's non-thinking mode, with the same architecture and tokenizer. Much better at tool use (BFCL-v3 61.9 vs. 57.6, TAU1-Retail 48.7 vs. 24.3). It never emits `<think>` blocks, so training, inference and the Ollama template all render tool calls the same way.
- Same Qwen3 family → shared tokenizer → logit-level KD stays open for v2.
- The baseline for G2 is the untrained **Qwen3-4B-Instruct-2507** itself.
- Apache 2.0 → clean for MIT release.
- **DoRA:** LoRA A/B matrices update direction; a separate trainable vector updates magnitude. Closer to full fine-tuning at low rank. One flag in PEFT (`use_dora=True`).
- **FSDP, not DeepSpeed ZeRO-2** — known QDoRA issues with ZeRO-2.

> **Open decision:** logprob capture is easier under vLLM than llama.cpp. Confirm the serving stack can return top-20 logprobs for the GGUF teacher, or serve the teacher via vLLM.

## 4. Hackathon focus areas

| Focus | What we will do on the cluster |
|---|---|
| **Distributed teacher inference** | Batch generation with Qwen3-32B Q4_K_M on one GPU (more if available). Start with ~10k examples, evaluate, then scale up if time allows. Measure throughput (tokens/s per GPU) and cost per 1k verified examples. |
| **Parallel student training** | QDoRA on Qwen3-4B-Instruct-2507 with FSDP; several seeds and ablations in parallel. A student family (1.7B / 4B) is a stretch goal (no 2507 release of 1.7B: use Qwen3-1.7B with thinking off). |
| **Accuracy vs. size under quantization** | Evaluate every (student size × quant level) pair on the same held-out set. Produce an **accuracy–size–speed Pareto frontier**. |

### Compute

- **Cluster:** Leonardo (CINECA). **One node (4× A100 64 GB), shared across hackathon teams.** Plan for 1–2 GPUs at a time.
- **Teacher:** Qwen3-32B Q4_K_M fits on a single 64 GB A100.
- **Student:** Qwen3-4B-Instruct-2507 QDoRA is small; FSDP gives clean multi-GPU scaling when GPUs are free.
- **Storage:** model weights + data + top-20 logprobs (can be several GB — check quota).
- **Operating rules on a shared node:** generate in chunks, checkpoint generated data frequently, run heavy jobs off-peak where possible.
- **Still to confirm:** compute-node access date; internet on compute nodes; vLLM availability; max wall time; storage quota.
- **If compute nodes have no internet:** pre-download weights to shared storage from login nodes; host the judge on the cluster.

## 5. Goals

### Must-have (definition of done)

- **G1 — End-to-end pipeline:** one config file runs generate → verify → train → quantize → evaluate for the HPC task, reproducibly, via Slurm job scripts.
- **G2 — A better small model:** the distilled student beats its own non-distilled base on the held-out HPC eval set, on executable-check pass rate, LLM-judge score and tool-call validity.
- **G3 — Runs locally:** the quantized student (+ RAG) runs on a laptop CPU in < 4 GB RAM, installable with one `ollama` command.
- **G4 — Open release:** a public MIT repo with docs, a model card, and a "bring your own teacher/task" guide.

### Should-have

- **G5 — Verified-data result:** whether filtering (executable checks + judge) beats unfiltered data in this domain.
- **G6 — Pareto frontier:** accuracy / size / tokens-per-second across ≥ 2 student sizes and ≥ 3 quant levels.
- **G7 — Framework generality:** distil a second student family or use a second teacher with config changes only.

### Stretch

- **G8:** RAFT-trained student + RAG vs. closed-book student.
- **G9:** DPO stage using verifier-derived chosen/rejected pairs.
- **G10:** Publish the verified synthetic HPC dataset (permissively licensed sources only) on the Hugging Face Hub.
- **G11:** Multi-node FSDP training of a student family (1.7B / 4B), with scaling efficiency reported.
- **G13:** Logit-level KD (KL + cross-entropy) using the saved top-20 logprobs.
- **G14:** Curriculum ordering by difficulty (pedagogical distillation, ICLR 2026).

## 6. Order of work

1. **Eval set first** — it defines the task.
2. **Tool schemas** — 5–10 HPC tools with names, parameters, return types.
3. **Gold examples** — hand-craft one prose answer and one prose + tool-call trace (question → reasoning → tool call → tool result → final answer). Used as few-shot anchors so teacher output stays consistently formatted.
4. **Generate ~10k** — ~70% prose, ~30% tool-calling traces. Coverage over volume: every tool, varied phrasings.
5. **Train → evaluate → generate more of what the student fails on.** Scale up only if time permits.

## 7. Success metrics

*Targets are provisional. We fix the exact numbers after we measure the baselines in week 1.*

| Metric | Baseline | Target |
|---|---|---|
| Executable-check pass rate (Slurm/CUDA/MPI answers) | base student, zero-shot | +20 pp over base |
| LLM-judge win rate vs. base student (pairwise, mean over 3 seeds) | 50% | > 60% |
| Gap to teacher (judge score) | teacher = 100% | student reaches ≥ 80% of teacher |
| Tool-call validity (correct structure + arguments) | base student | +20 pp over base |
| Hallucinated Slurm flags per answer | base student | −50% |
| Q4_K_M vs. bf16 accuracy drop | — | ≤ 2 pp |
| Laptop memory (Q4_K_M, model + RAG) | — | < 4 GB |
| Laptop CPU speed | — | ≥ 20 tokens/s |
| Verified examples produced | — | ≥ 10k (scale up if throughput allows) |

### Eval set (~150–200 questions, human-reviewed, strictly held out)

- **Prose:** LLM-judge scoring of Slurm/CUDA/MPI explanations.
- **Executable:** sbatch scripts and CUDA snippets must parse/compile.
- **Tool calling:** structurally valid call with correct arguments; also tests *when* to call vs. answer directly.

### Evaluation integrity rules (non-negotiable)

- Split the data **by source document** before any generation.
- Human-review the core eval set.
- Decontaminate training data against the eval set.
- Report negative results as well as positive ones.
- Report every score as **mean ± spread over at least 3 training seeds**.

## 8. Expected outcomes / deliverables

1. **`distillkit` Python package** (MIT), installable with `pip install distillkit[gen,train,export]`, driven by a CLI and YAML configs.
2. **HPC Assistant model:** HF weights + GGUF (Q8_0, Q4_K_M) + Ollama Modelfile + model card with eval results.
3. **Evaluation harness and HPC eval set:** questions tagged by topic and check type, with executable and tool-call checkers.
4. **Cluster recipes:** Slurm scripts for teacher generation and parallel multi-seed QDoRA training on a shared Leonardo node, with measured throughput.
5. **Teacher logprob dataset:** top-20 logprobs per token, ready for v2 logit-level KD.
6. **Results report:** ablations (filtering, data scale, student size, quantization) and the Pareto frontier plot.
7. **Final presentation / demo:** live laptop demo of the HPC Assistant answering, calling tools and writing a Slurm script offline, next to the base model.

## 9. Non-goals

- Logit-level / on-policy distillation *training* in v1 (logprobs are saved; training is v2).
- Training or pre-training a model from scratch.
- A polished GUI; CLI plus Ollama is enough for the demo.
- Beating frontier models in general. We target this domain at this size.

## 10. Team roles

| Workstream | Owns |
|---|---|
| **Data & seeds** | corpus collection, chunking, provenance, licensing, train/eval document split |
| **Teacher inference** | teacher serving on the cluster, logprob capture, throughput, prompt templates, gold examples, diversity grid |
| **Verification & eval** | executable checkers, tool-call checker, LLM judge, dedup/decontamination, eval harness, human-reviewed eval set |
| **Student training** | QDoRA + FSDP, hyperparameters, ablation runs |
| **Export, RAG & release** | GGUF/imatrix, Ollama packaging, local RAG, laptop benchmarks, docs, demo |

## 11. Timeline

| When | Milestone |
|---|---|
| **Before Oct 6** | Seed corpus + document split; eval set started (~50 questions incl. tool-calling slice); tool schemas + 2 gold examples; toy pipeline running locally. |
| **Week 1 (Oct 6–12)** | Kickoff; Leonardo access; pre-download teacher (Qwen3-32B Q4_K_M) and student (Qwen3-4B-Instruct-2507); smoke test of teacher generation with logprobs and QDoRA training + GGUF export; baselines measured; 10k pilot generation |
| **Week 2 (Oct 13–19)** | Train first student; evaluate; targeted second generation batch on failure areas; verification at scale |
| **Week 3 (Oct 20–26)** | Ablations (3 seeds each): filtering, data scale; quant levels; second student size if on track |
| **Final (Oct 27–29)** | Quantization sweep + Pareto frontier; release repo, model and dataset; final presentation |

**Checkpoint rule:** if G1 (the end-to-end pipeline) isn't green by **Oct 16**, freeze stretch goals and put everyone on the must-haves.
