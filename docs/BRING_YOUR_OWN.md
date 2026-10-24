# Bring your own teacher, student or task

DistillKit is not tied to Qwen or to HPC. This guide shows how to point it at another teacher, another student or a
new domain, using only config changes. The HPC assistant in this repository is one worked example; the lessons from
it are at the end.

## 1. What you need

- A **teacher**: any model behind an OpenAI-compatible `/chat/completions` endpoint (vLLM, llama-server, Ollama, or a
  hosted API).
- A **student**: any causal LM on Hugging Face that TRL/PEFT can fine-tune (LoRA or DoRA).
- A **judge**: ideally from a different model family than the teacher and student, so it does not favour their style.
- **Seed documents** for your domain and a held-out **evaluation set**.

## 2. Swap the teacher

The `teacher` block of the config is all that changes:

```yaml
teacher:
  base_url: http://<node>:8000/v1     # vllm serve <model>; or http://localhost:11434/v1 for Ollama
  api_key_env: null                   # name of the env var with the key, null for local servers
  model: Qwen/Qwen3-32B-AWQ           # the name the server exposes
  temperature: 0.7
  max_tokens: 1024
  concurrency: 12
  top_logprobs: 20                    # optional: save the teacher's top-20 logprobs for logit distillation
```

- **Sequence-level distillation** (learning from the teacher's text) works with any teacher, whatever its tokenizer.
- **Logit distillation** (`top_logprobs: 20`, then `research/kd/train_kd.py`) needs a teacher with the **same
  vocabulary** as the student. In this project every teacher was a Qwen3 model (Qwen3-32B, Qwen3-Next-80B,
  Qwen3-235B) and the student Qwen3-4B, so the distributions line up token for token. It was our largest single gain.
- Very large teachers load much faster from node RAM than from a parallel file system: see
  `research/kd/score235.sbatch` (copy to `/dev/shm`, then start vLLM; 764 s down to 43 s for a 46 GB checkpoint).

## 3. Swap the student

```yaml
student:
  model: Qwen/Qwen3-4B-Instruct-2507   # any HF causal LM
  system_prompt: You are an HPC assistant. ...
train:
  method: lora                         # or dora
  lora_r: 16
  lora_alpha: 32
  learning_rate: 1.0e-4
  epochs: 1
```

Keep the `system_prompt` identical in training, evaluation and the exported Modelfile. A smaller student is cheaper
but can fall off sharply: in this project Qwen3-1.7B scored 19 points lower than Qwen3-4B on the same data.

## 4. A new domain or task

1. **Seed documents.** Put Markdown or text files in a folder and set `seeds.dir`. Each file is chunked
   (`seeds.chunk_chars`), and questions are generated from the chunks, so every answer can be checked against its
   source. Only use sources whose licence allows it, and record the licence per document.
2. **Hold out evaluation documents before generating anything.** List them in `seeds.eval_docs`; no question is
   generated from them.
3. **Evaluation set.** A JSONL file with one question per line:
   `{"id": "arr-001", "doc_id": "...", "task": "script", "question": "...", "reference": "..."}`. Point `eval.file` at
   it. Tool-calling questions go in `eval.tool_file`
   (`{"id", "question", "expect": "call", "expected_call": {"name", "arguments"}}`).
4. **Question variety.** `generate.personas` and `generate.task_types` set who asks and what kind of question (concept,
   how-to, script, debug). `generate.tool_fraction` and `generate.tool_mix` control tool-calling traces.
5. **Tools.** The tool schemas live in `src/distillkit/tools.py` (`SCHEMAS`), with a mock environment for the tool
   results. Replace them with your domain's tools; the verifier checks every generated call against the schemas.
6. **Domain checks.** `verify.flag_list` points at the list of valid command-line flags (Slurm's by default); replace it
   or extend `src/distillkit/checks.py` with checks that fit your domain.

## 5. Run it

```bash
uv sync --extra train --extra export
uv run distillkit generate -c configs/my_task.yaml   # teacher writes questions, answers, tool traces
uv run distillkit verify   -c configs/my_task.yaml   # dedup, checks, tool-call validation, decontamination
uv run distillkit train    -c configs/my_task.yaml   # LoRA/DoRA fine-tuning of the student
uv run distillkit evaluate -c configs/my_task.yaml   # student vs base: pairwise judge + tool-call scoring
uv run distillkit export   -c configs/my_task.yaml   # merge, GGUF, importance matrix, Q4_K_M/Q8_0, Ollama Modelfile
uv run distillkit bench    -c configs/my_task.yaml   # laptop speed, peak RAM, time to first token
```

Each stage appends to `run_dir` and skips finished work, so it can be rerun after a crash or a wall-time kill. For a
cluster, the Slurm scripts in `slurm/` and `research/` show the same stages as batch jobs.

## 6. Evaluate honestly

- Calibrate the judge first (`distillkit calibrate` on known-label pairs).
- Judge pairwise **in both answer orders**, and use two or three judges from different families with a majority vote.
- Compare only numbers from the **same judge run**; runs drift by 2-4 points.
- Train 3 seeds and report all of them; check a gain with a paired confidence interval
  (`research/rag65/paired.py`).
- Give every model the same answer length budget (`research/exp30/budget768.py`): judges favour long answers.

## 7. What worked in the HPC example

| Worked | Did not work |
|---|---|
| Logit distillation from a same-vocabulary teacher | DPO on verifier-derived pairs |
| On-policy distillation (the student's own answers, scored by the teacher) | Mixing answers from different teachers |
| A uniform weight soup of several adapters (`research/round27/soup_full.py`) | Plain fine-tuning on top of a souped model |
| A better retriever at inference (`research/rag65/retrieve.py`, `RAG_VARIANT=b`) | A smaller student (1.7B) |
| Relevant questions over generic volume | More training rounds once the 4B model plateaued |

The full history, with numbers, is in the Hugging Face model card of
[VaibhavKKM/hpc-assistant-qwen3-4b](https://huggingface.co/VaibhavKKM/hpc-assistant-qwen3-4b).
