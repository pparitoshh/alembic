# How training works (plain-language guide)

*For anyone new to this repo. No ML background assumed. Code: [`src/distillkit/train.py`](../src/distillkit/train.py), settings: `train:` section of [`configs/toy.yaml`](../configs/toy.yaml). Run it with `uv run distillkit train -c configs/toy.yaml`.*

## The idea in one paragraph

We have a **big, smart "teacher" model** (reached over an API) and a **small "student" model** (Qwen3-0.6B, runs on a laptop GPU). The teacher has already written answers to Slurm/HPC questions (`runs/toy/verified.jsonl`). Training means showing the student those question → answer pairs over and over, and nudging it each time so that its own answers look more like the teacher's. This is called **distillation**: copying a big model's know-how into a small one.

---

## 1. What one training sample looks like

Each row in `verified.jsonl` becomes one sample with two parts: a **prompt** (what the student is given) and a **completion** (what it should learn to write).

Here is a real sample from our toy data:

**Prompt** (69 tokens) — the system instructions + the question, wrapped in Qwen's chat format:

```text
<|im_start|>system
You are an HPC assistant. Answer questions about Slurm, CUDA, MPI and profiling concisely and correctly. Put scripts in fenced code blocks.<|im_end|>
<|im_start|>user
How do I start an interactive shell directly on a compute node for 30 minutes using one task?<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**Completion** (26 tokens) — the teacher's answer, plus an "end of text" marker so the student learns when to stop:

````text
```bash
srun --pty --time=00:30:00 --ntasks=1 bash
```<|im_end|>
````

Notes:
- `<|im_start|>` / `<|im_end|>` are Qwen's markers for "a message starts/ends here".
- The empty `<think></think>` means **thinking mode is off**: we want short, direct answers, not long reasoning. The same prompt format is used in training and evaluation (`src/distillkit/prompting.py`), so the student never sees a mismatch.
- A **token** is a chunk of text the model reads and writes — roughly a word piece. The answer above is split into tokens like `` ``` ``, `bash`, `\n`, `s`, `run`, ` --`, `pty`, ` --`, `time`, `=`, `0`, `0`, …

## 2. What the student is actually predicting

A language model does exactly one thing: **given the text so far, guess the next token.**

During training we feed the student the prompt plus the teacher's answer, and at every position in the answer we ask: *"what comes next?"*

| Student has seen … | It should predict |
|---|---|
| prompt | `` ``` `` |
| prompt + `` ``` `` | `bash` |
| prompt + `` ```bash `` | `\n` |
| prompt + `` ```bash\n `` | `s` |
| prompt + `` ```bash\ns `` | `run` |
| … | … |
| prompt + whole answer | `<|im_end|>` (stop) |

For each guess, the model outputs a probability for every one of its ~152,000 possible tokens. The **loss** measures how much probability it gave to the *correct* token (the teacher's): high probability → low loss, low probability → high loss.

Two important details:
- **The student always sees the teacher's real previous tokens**, not its own guesses. So one sample gives us 26 separate "guess the next token" exercises in a single pass.
- **Only the answer is scored.** We don't care whether the student can predict the question or the system prompt — it will always be *given* those. So loss is counted on the 26 answer tokens only, not the 69 prompt tokens (`completion_only_loss=True`).

## 3. Micro-batches, gradient accumulation and optimizer steps

### The vocabulary

| Term | Plain meaning |
|---|---|
| **Gradient** | For every trainable number in the model: "which way, and how hard, should I nudge you to lower the loss?" |
| **Micro-batch** | How many samples the GPU processes at once. Ours: **4** (`batch_size`). Limited by the 6 GB of GPU memory. |
| **Gradient accumulation** | Instead of changing the model after every micro-batch, we **add up** the gradients of several micro-batches first. Ours: **4** (`grad_accum`). |
| **Effective batch** | Samples per weight update = 4 × 4 = **16**. |
| **Optimizer step** | The moment the model's weights actually **change**, using the added-up gradients. |
| **Epoch** | One full pass over all training samples. We do **2** (`epochs`). |
| **Learning rate** | How big each nudge is. Ours peaks at **0.0002** and then slowly decreases (cosine schedule). |

### What happens inside one micro-batch (weights do NOT change)

1. **Forward pass** — the student reads the 4 samples and makes all its next-token guesses.
2. **Loss** — score those guesses against the teacher's answers.
3. **Backward pass** — compute the gradient for every trainable number.
4. **Add** these gradients to a running total.

### What happens in an optimizer step (weights DO change)

1. Take the running total of gradients (averaged over the 16 samples).
2. The **optimizer (AdamW)** nudges every trainable number: roughly *new value = old value − learning rate × gradient* (AdamW also smooths the nudges using recent history, and gently pulls values toward zero so they don't grow too large).
3. Reset the running total to zero.

### One epoch over our 40 toy samples

```
micro-batch  samples   what happens
    1         1–4      guess, score, compute gradients → store
    2         5–8      same → add to stored
    3         9–12     same → add to stored
    4        13–16     same → add → OPTIMIZER STEP 1 (weights change), reset
    5        17–20     ┐
    6        21–24     │
    7        25–28     │
    8        29–32     ┘ → OPTIMIZER STEP 2
    9        33–36     ┐
   10        37–40     ┘ → OPTIMIZER STEP 3  (only 8 samples left, still counts)
```

So one epoch = **10 micro-batches but only 3 weight updates**. Two epochs = **6 updates** in total.

**Why not just use micro-batches of 16?** 16 samples at once don't fit in 6 GB of GPU memory. Accumulating 4 × 4 gives almost the same update as a real batch of 16 (bigger batches = steadier, less noisy nudges), at the memory cost of 4. It just takes a bit longer.

## 4. Which weights are frozen and which are trained

### The student's anatomy

Qwen3-0.6B is a stack of **28 identical layers**. Text flows through them one by one:

```
text → [token embeddings] → layer 1 → layer 2 → … → layer 28 → [output head] → next-token probabilities
```

Each layer has two parts, and each part is mostly made of big grids of numbers called **linear layers** (weight matrices):

| Part | What it does (roughly) | Its linear layers |
|---|---|---|
| **Attention** | Lets each token "look back" at earlier tokens to gather context (e.g. `--time` connects to "30 minutes" in the question) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **MLP (feed-forward)** | Where much of the model's stored knowledge lives; transforms each token's information | `gate_proj`, `up_proj`, `down_proj` |

Plus some small pieces: **layer norms** (keep numbers in a healthy range), the **token embeddings** (turn tokens into numbers) and the **output head** (turn numbers back into token probabilities — in Qwen3-0.6B this shares its weights with the embeddings).

### Frozen: all ~596 million original weights

**Every original weight of Qwen3-0.6B is frozen** — it is never changed during training. That includes:
- all attention and MLP matrices in all 28 layers,
- the token embeddings and output head,
- all layer norms.

Why freeze? It keeps the student's general language ability intact (hard to "break" the model), needs far less GPU memory, and makes training fast.

### Trained: small LoRA add-ons (~40 million numbers, 6.3%)

We use **LoRA** (Low-Rank Adaptation). Next to each frozen linear layer we attach a small, trainable **"correction"** made of two thin matrices, **A** and **B**. The layer's output becomes:

```
output = frozen original layer(input)  +  2 × B(A(input))
                  ↑ never changes           ↑ LoRA: the only thing trained
```

(The `2×` comes from `lora_alpha / lora_r` = 128 / 64.)

Concrete sizes for one attention layer, `q_proj` in layer 1:

| | Shape | Numbers | Trained? |
|---|---|---|---|
| Original `q_proj` weight | 2048 × 1024 | ~2.1 million | ❄️ frozen |
| LoRA **A** | 64 × 1024 | ~65 thousand | 🔥 trained |
| LoRA **B** | 2048 × 64 | ~131 thousand | 🔥 trained |

The **64** is the LoRA **rank** (`lora_r`): how "thick" the correction is. Bigger rank = more capacity to learn, but more memory.

LoRA is attached to **all 7 linear layer types** (`target_modules="all-linear"`) in **all 28 layers** — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — so 196 LoRA add-ons in total. The output head is excluded.

| | Numbers |
|---|---|
| Whole model incl. LoRA | 636,420,096 |
| Frozen (original Qwen3-0.6B) | ~596 million (93.7%) |
| **Trained (LoRA A + B)** | **40,370,176 (6.3%)** |

Analogy: the original model is a printed textbook we are not allowed to edit. LoRA is a set of sticky notes we attach to each page. Training only rewrites the sticky notes; the book stays as it is.

At the start of training, every **B** is all zeros, so the add-ons do nothing and the student behaves exactly like the original model. Training gradually fills them in.

## 5. Settings that make it fit on a 6 GB laptop GPU (RTX 2060)

- **fp16 mixed precision** — the heavy maths runs in 16-bit numbers (fast, half the memory), while the master copy of the weights stays in 32-bit so tiny nudges aren't rounded away. (Newer GPUs would use bf16; the RTX 2060 doesn't support it.)
- **Gradient checkpointing** — instead of keeping every intermediate result for the backward pass, the model recomputes some of them. A little slower, much less memory.
- **Max length 1024 tokens** — longer samples are cut off. Our answers are far shorter.
- **`TORCH_DISABLE_NATIVE_JIT=1`** (set in `src/distillkit/cli.py`) — torch 2.14 otherwise tries to compile a special GPU kernel that needs a C compiler, which this machine doesn't have.

## 6. What comes out

Only the trained LoRA add-ons are saved, to **`runs/toy/adapter/`** (~160 MB; the original model is downloaded separately and never modified).

- **`evaluate`** loads the original model + this adapter and compares it with the original model alone.
- **`export`** (planned) merges the add-ons permanently into the model and converts it for Ollama.

## 7. What the first toy run looked like (2026-09-30)

40 samples, 2 epochs, 6 optimizer steps, 28 seconds of training:

| Step | Epoch | Loss | Tokens guessed right |
|---|---|---|---|
| 1 | 0.4 | 1.85 | 59% |
| 2 | 0.8 | 1.83 | 61% |
| 3 | 1.0 | 1.55 | 64% |
| 4 | 1.4 | 1.47 | 66% |
| 5 | 1.8 | 1.34 | 69% |
| 6 | 2.0 | 1.33 | 69% |

How to read this:
- **Loss going down** = the student is getting better at predicting the teacher's answers. ✅
- On only 40 samples, most of that is **memorising** them. It proves the pipeline works; it does **not** prove the student got smarter. That is what the held-out evaluation checks.
- Step 1 had learning rate 0 (the "warmup" phase starts at zero), so it changed nothing. This doesn't matter with realistic data sizes.
- The real run (~5,000 samples, Qwen3-1.7B, on the cluster) uses exactly the same code with a different config.

## Glossary

| Term | Meaning |
|---|---|
| Teacher | Large model that writes the training answers (`qwen3.8-flash` via API) |
| Student | Small model we train (`Qwen/Qwen3-0.6B` locally, `Qwen3-1.7B` on the cluster) |
| Token | A word piece; the unit a model reads and writes |
| Loss | A score for how wrong the guesses were; lower is better |
| Gradient | Direction and size of the nudge for each trainable number |
| Micro-batch | Samples processed by the GPU at once (4) |
| Gradient accumulation | Adding up gradients over several micro-batches before updating (4) |
| Optimizer step | One actual update of the trainable weights |
| Epoch | One pass over the whole training set |
| Learning rate | Size of each nudge |
| LoRA | Small trainable add-ons next to frozen layers |
| Rank (`r`) | Thickness of the LoRA add-ons (64) |
| Adapter | The saved LoRA add-ons (`runs/toy/adapter/`) |
