---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
tags: [gguf, ollama, slurm, hpc]
---

# hpc-assistant x0.2 (GGUF, Q4_K_M)

A Slurm/HPC assistant made from Qwen3-4B-Instruct-2507 (Alembic team, EuroHPC hackathon, October 2026).

## How the model was made

1. Fine-tuning (B7): DoRA on 20,472 Slurm/HPC examples written by Qwen3-32B-AWQ (teacher model), seed 42.
2. Scaling: only 20% of the trained change is kept: `W = W_base + 0.2 × (W_B7 − W_base)`. No new training.
3. Export: llama.cpp (tag b11392) GGUF, Q4_K_M quantization without an importance matrix.

## Results (judge gpt-oss-20b, paired orders, win rate against the base model, mean of 3 seeds)

| Test set | Win rate |
|---|---|
| v3: 200 questions on trained topics | 0.528 |
| v2: 66 questions on topics that are not in the training data | 0.481 |

0.5 is equal to the base model. The win on v3 is small. The scale 0.2 was chosen with the v3 results (development result).
In a manual check of 20 questions, the model was not better than the base model. The model can invent Slurm options:
check each command before you use it.

## Use with Ollama

```bash
ollama run hf.co/gparitosh007/hpc-assistant-x0.2-gguf:Q4_K_M
```

Or download the files and run `ollama create hpc-assistant-x02 -f Modelfile`.

This is a research model from a hackathon. It is not a supported product. Do not run its commands on a
production cluster without a check.

The `Modelfile` (local use) and the files `template`, `system` and `params` (for `ollama run hf.co/...`) contain the
Qwen3 tool-calling template of the evaluation. The system prompt adds the model name ("HPC Assistant"); the evaluation
used the same prompt without the first sentence.
The evaluation used greedy decoding (temperature 0). The Modelfile uses temperature 0.7.

sha256 of `model-Q4_K_M.gguf`: `263c1204e421f27b0bbf7bf0d0ded04f6fa5dc2fba5d9b923e713f04490136d6`
