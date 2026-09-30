# DistillKit — Goals & Expected Outcomes

*European AI Hackathon · Oct 6–29, 2026 · Team of 5*
*Companion to [RESEARCH.md](RESEARCH.md), which covers the state of the art and the design rationale.*

---

## 1. Mission

**Make large open-weight models practical to run outside the data centre.**

We use the cluster once, to turn a large teacher model into a small, specialised student. The student then runs forever on a laptop: private, offline, and free.

## 2. The problem

- **For developers:** existing distillation scripts are tied to specific architectures and often assume a shared tokenizer. There is no clean, reusable package that takes *any* teacher and *any* downstream task and produces a laptop-sized model plus honest evidence of how good it is.
- **For HPC users:** people on clusters constantly ask the same Slurm, CUDA, MPI and profiling questions. General chatbots hallucinate flags and APIs, cloud assistants are often not allowed on sensitive clusters, and there is no private, offline assistant for cluster work.

## 3. What we are building

An **MIT-licensed, tokenizer-independent, sequence-level distillation framework** (the student learns only from teacher-generated Q&A text; no logits). It has four pluggable modules:

1. **Teacher** — any model that emits text (a Hugging Face model via vLLM, or an OpenAI-compatible endpoint).
2. **Data generation** — document-grounded, diversified question generation; several teacher answers per question; **verification** (dedup, executable checks, LLM judge, decontamination).
3. **Student training** — TRL SFT (full fine-tuning or LoRA), multi-node with FSDP/DeepSpeed; optional RAFT and DPO stages.
4. **Quantize, export and evaluate** — GGUF (Q8_0 / Q4_K_M with imatrix), an Ollama package, and an evaluation harness run at every size and quantization level.

**Demo use case: an HPC Assistant**, a ≤2B student specialised in Slurm, CUDA, MPI and profiling, shipped with a small local RAG index over the same documentation.

## 4. Hackathon focus areas

These map directly to the team's stated focus.

| Focus | What we will do on the cluster |
|---|---|
| **Distributed teacher inference at scale** | Multi-GPU / multi-node vLLM batch generation of about 50–100k questions × 4–8 answers each. Measure throughput (tokens/s per GPU) and cost per 1k verified examples. |
| **Multi-node student training** | TRL + FSDP/DeepSpeed across nodes. Train a *family* of students (0.8B / 2B / 4B) and the ablation runs in parallel. Report scaling efficiency. |
| **Accuracy vs. size under quantization** | Evaluate every (student size × quant level) pair on the same held-out set. Produce an **accuracy–size–speed Pareto frontier**. |

## 5. Goals

### Must-have (definition of done)

- **G1 — End-to-end pipeline:** one config file runs *generate → verify → train → quantize → evaluate* for the HPC task, reproducibly, via Slurm job scripts.
- **G2 — A better small model:** the distilled student beats its own non-distilled base model on our held-out HPC eval set, on both the executable-check pass rate and the LLM-judge score.
- **G3 — Runs locally:** the quantized student (+ RAG) runs on a laptop CPU in **< 4 GB RAM**, installable with one `ollama` command.
- **G4 — Open release:** a public MIT repo with docs, a model card, and a "bring your own teacher/task" guide.

### Should-have

- **G5 — Verified-data result:** a clear answer on whether filtering (executable checks + judge) beats unfiltered data in this domain.
- **G6 — Pareto frontier:** an accuracy / size / tokens-per-second plot across ≥ 2 student sizes and ≥ 3 quant levels.
- **G7 — Framework generality:** the same pipeline distils a *second* student family (e.g. Gemma), or uses a second teacher, with config changes only.

### Stretch

- **G8:** RAFT-trained student + RAG vs. closed-book student.
- **G9:** DPO stage using verifier-derived chosen/rejected pairs.
- **G10:** Publish the verified synthetic HPC dataset (permissively licensed sources only) on the Hugging Face Hub.

## 6. Success metrics

Targets are provisional. We fix the exact numbers after we measure the baselines in week 1.

| Metric | Baseline | Target |
|---|---|---|
| Executable-check pass rate (Slurm/CUDA/MPI answers) | base student, zero-shot | **+20 pp** over base |
| LLM-judge win rate vs. base student (pairwise) | 50% | **≥ 70%** |
| Gap to teacher (judge score) | teacher = 100% | student reaches **≥ 80%** of the teacher score |
| Hallucinated Slurm flags per answer | base student | **−50%** |
| Q4_K_M vs. bf16 accuracy drop | — | **≤ 2 pp** |
| Laptop memory (Q4_K_M, model + RAG) | — | **< 4 GB** |
| Laptop CPU speed | — | **≥ 20 tokens/s** |
| Verified examples produced | — | **≥ 50k** |

**Evaluation integrity rules** (non-negotiable):
- Split the data by source document *before* any generation.
- Human-review the core eval set.
- Decontaminate training data against the eval set.
- Report negative results as well as positive ones.

## 7. Expected outcomes / deliverables

1. **`distillkit` Python package** (MIT), installable with `pip install distillkit[gen,train,export]`, driven by a CLI and YAML configs.
2. **HPC Assistant model:** HF weights + GGUF (Q8_0, Q4_K_M) + Ollama `Modelfile` + model card with eval results.
3. **Evaluation harness and HPC eval set:** questions tagged by topic and check type, with the executable checkers.
4. **Cluster recipes:** Slurm scripts for distributed vLLM generation and multi-node FSDP/DeepSpeed training, with the throughput numbers we measured.
5. **Results report:** ablations (filtering, answers per question, teacher choice, data scale, student size, quantization) and the Pareto frontier plot.
6. **Final presentation / demo:** a live laptop demo of the HPC Assistant answering and writing a Slurm script offline, next to the base model.

## 8. Non-goals

- Logit-level / on-policy distillation (a v2 candidate; see RESEARCH.md §1).
- Training or pre-training a model from scratch.
- A polished GUI; the CLI plus Ollama is enough for the demo.
- Beating frontier models in general. We target *this domain at this size*.

## 9. Team roles

We have five people. Each workstream has one owner, and everyone reviews each other's work.

| Workstream | Owns |
|---|---|
| **Data & seeds** | corpus collection, chunking, provenance, licensing, train/eval document split |
| **Teacher inference** | vLLM on the cluster, generation throughput, prompt templates, diversity grid |
| **Verification & eval** | executable checkers, LLM judge, dedup/decontamination, eval harness, human-reviewed eval set |
| **Student training** | TRL + FSDP/DeepSpeed multi-node, hyperparameters, ablation runs |
| **Export, RAG & release** | GGUF/imatrix, Ollama packaging, local RAG, laptop benchmarks, docs, demo |

## 10. Timeline

| When | Milestone |
|---|---|
| **Before Oct 6** | Seed corpus collected + document split; ~50 hand-written eval questions; toy-scale pipeline running locally |
| **Week 1 (Oct 6–12)** | Kickoff; cluster access; **day-1 smoke test** of the student architecture (train + GGUF); baselines measured; 5k-example pilot generation |
| **Week 2 (Oct 13–19)** | Full-scale teacher generation (multi-sample); verification pipeline at scale; first trained student |
| **Week 3 (Oct 20–26)** | Multi-node training of the student family + ablations; RAFT/DPO if on track |
| **Final (Oct 27–29)** | Quantization sweep + Pareto frontier; release repo, model and dataset; final presentation |

**Checkpoint rule:** if G1 (the end-to-end pipeline) isn't green by Oct 16, freeze stretch goals and put everyone on the must-haves.
