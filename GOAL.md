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
3. **Student training** — TRL SFT (LoRA or full fine-tuning) on a single GPU per run; multi-node FSDP/DeepSpeed and RAFT/DPO stages are optional.
4. **Quantize, export and evaluate** — GGUF (Q8_0 / Q4_K_M with imatrix), an Ollama package, and an evaluation harness run at every size and quantization level.

**Demo use case: an HPC Assistant**, a ≤2B student specialised in Slurm, CUDA, MPI and profiling, shipped with a small local RAG index over the same documentation.

## 4. Hackathon focus areas

These map directly to the team's stated focus.

| Focus | What we will do on the cluster |
|---|---|
| **Distributed teacher inference** | Multi-GPU vLLM batch generation with a ~30B teacher (tensor parallel over 2 GPUs, 2 replicas on one node). Start with ~5k verified examples and scale up if time allows. Measure throughput (tokens/s per GPU) and cost per 1k verified examples. |
| **Parallel student training** | TRL on a single GPU per run (LoRA or full fine-tune), with several seeds and ablations running in parallel on the second node. Multi-node FSDP/DeepSpeed and a student *family* (0.6B / 1.7B / 4B) are stretch goals. |
| **Accuracy vs. size under quantization** | Evaluate every (student size × quant level) pair on the same held-out set. Produce an **accuracy–size–speed Pareto frontier**. |

### Compute (requested from the mentor, Sep 30)

- **Cluster:** Leonardo (CINECA). Requested **2 nodes (8× A100 64 GB)**; the minimum is **1 node (4 GPUs)**.
- **Node 1:** vLLM serving the teacher to generate training data. This is the most GPU-heavy step.
- **Node 2:** student fine-tuning (Qwen3-1.7B, one GPU per run, several seeds in parallel) and LLM-judge evaluation.
- **Storage:** about 150 GB for model weights and data (the teacher alone is ~65 GB).
- **Asked about, waiting for answers:** compute-node access from Oct 6 (Oct 9 at the latest); internet on compute nodes (Hugging Face downloads, outbound HTTPS to the judge API); whether vLLM is available as a module/container; max job wall time; storage quota.
- **If compute nodes have no internet:** pre-download the weights to shared storage from the login nodes, and host the judge on the cluster with vLLM.
- **If only 1 node is granted:** 2 GPUs for the teacher (1 replica), 2 GPUs for training and eval.

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
- **G11:** Multi-node FSDP/DeepSpeed training of a student family (0.6B / 1.7B / 4B), with scaling efficiency reported.

## 6. Success metrics

Targets are provisional. We fix the exact numbers after we measure the baselines in week 1.

| Metric | Baseline | Target |
|---|---|---|
| Executable-check pass rate (Slurm/CUDA/MPI answers) | base student, zero-shot | **+20 pp** over base |
| LLM-judge win rate vs. base student (pairwise, mean over 3 seeds) | 50% | **> 60%** (toy run: 64.1% ± 4.8) |
| Gap to teacher (judge score) | teacher = 100% | student reaches **≥ 80%** of the teacher score |
| Hallucinated Slurm flags per answer | base student | **−50%** |
| Q4_K_M vs. bf16 accuracy drop | — | **≤ 2 pp** |
| Laptop memory (Q4_K_M, model + RAG) | — | **< 4 GB** |
| Laptop CPU speed | — | **≥ 20 tokens/s** |
| Verified examples produced | — | **≥ 5k** (scale up if throughput allows) |

**Evaluation integrity rules** (non-negotiable):
- Split the data by source document *before* any generation.
- Human-review the core eval set.
- Decontaminate training data against the eval set.
- Report negative results as well as positive ones.
- Report every score as a mean ± spread over at least 3 training seeds. On the toy eval, seeds differ by about ±5 points (see [PROGRESS.md](PROGRESS.md)).

## 7. Expected outcomes / deliverables

1. **`distillkit` Python package** (MIT), installable with `pip install distillkit[gen,train,export]`, driven by a CLI and YAML configs.
2. **HPC Assistant model:** HF weights + GGUF (Q8_0, Q4_K_M) + Ollama `Modelfile` + model card with eval results.
3. **Evaluation harness and HPC eval set:** questions tagged by topic and check type, with the executable checkers.
4. **Cluster recipes:** Slurm scripts for multi-GPU vLLM generation and parallel multi-seed training on Leonardo, with the throughput numbers we measured.
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
| **Before Oct 6** | Seed corpus collected + document split; ~50 hand-written eval questions; toy-scale pipeline running locally. *Status Sep 30: toy pipeline generate → evaluate works and beats base (64.1% ± 4.8); export, real corpus and the 50-question eval set still open.* |
| **Week 1 (Oct 6–12)** | Kickoff; Leonardo access (asked for Oct 6, Oct 9 at the latest); pre-download teacher weights; **smoke test** of vLLM teacher and student training (train + GGUF); baselines measured; 5k-example pilot generation |
| **Week 2 (Oct 13–19)** | Full-scale teacher generation (multi-sample); verification pipeline at scale; first trained student |
| **Week 3 (Oct 20–26)** | Ablations (3 seeds each) and quant levels; second student size; RAFT/DPO/multi-node if on track |
| **Final (Oct 27–29)** | Quantization sweep + Pareto frontier; release repo, model and dataset; final presentation |

**Checkpoint rule:** if G1 (the end-to-end pipeline) isn't green by Oct 16, freeze stretch goals and put everyone on the must-haves.
