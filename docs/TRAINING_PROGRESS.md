# Training progress

A log of every student training batch: what was run (data, code, config, jobs) and what came out
(loss, judged quality). Newest batch last. Add a section per batch and fill in results as the
evaluation jobs finish; keep numbers as reported by the job outputs, not rounded guesses.

Where results come from (all paths relative to `$REPO` on Leonardo):

| Result | File | Written by |
|---|---|---|
| Train / validation loss per step | `seed_<n>/adapters/step_<N>/adapter/trainer_state.json`, MLflow | `slurm/train.sbatch` |
| Final adapter, config eval set | `seed_<n>/eval_summary.json` | `slurm/evaluate.sbatch` |
| Every kept step, dataset_v1 dev slice | `seed_<n>/adapters/step_<N>/eval_summary.json`, MLflow `dev/...` | `slurm/evaluate_checkpoints.sbatch` |

MLflow store: `$REPO/mlruns`, experiment `distillkit`, one run per seed named after its run dir
(view with `uvx mlflow ui`, see `slurm/README.md`).

## Batch 1: QDoRA on verified_qwen_4775 (2026-10-10)

### Setup

- **Data:** `data/teacher/verified_qwen_4775/verified.jsonl` (PR #25), 4,775 certified teacher
  records, SHA-256 `3f99eefb…`. Tokens per example: median 164, max 1,520; 0 over `max_length`.
- **Student:** `Qwen/Qwen3-4B-Instruct-2507`, QDoRA (`method: dora`, 4-bit).
- **Config:** `configs/qwen3_4b_qdora.yaml`

  | Setting | Value |
  |---|---|
  | lora_r / alpha / dropout | 16 / 32 / 0.0 |
  | learning rate | 1e-4, cosine |
  | epochs | 3 |
  | batch size x grad accum | 4 x 4 (effective 16) |
  | max_length | 2048 |
  | bf16 | yes |
  | save_steps | 100 (newest 2 checkpoints kept) |
  | val_fraction | 0.05 (seeds 1, 2 only) |

- **Hardware:** 1 x A100 per seed (Booster, `boost_usr_prod`), ~3.1 s/step.

### Runs

| Seed | Run dir | Code | Validation split | MLflow | Train job | Steps | Wall time |
|---|---|---|---|---|---|---|---|
| 42 | `runs/verified4775_qdora` | 5873df7 (PR #27) | none (4,775 train) | no | 59911612 | 897 | 51.5 min |
| 1 | `runs/verified4775_qdora_v2` | a15dc23 (PR #28) | 4,530 train / 245 val | yes | 59912914 | 852 | 47.8 min |
| 2 | `runs/verified4775_qdora_v2` | 3738b9b (PR #29) | 4,530 train / 245 val | yes | 59915034 | 852 | running |

Seed 42 ran on the older code: no validation split, no MLflow, and only the last two checkpoints
survive (steps 800, 897), so its per-step dev eval covers just those two.

### Adapter locations

All under `$REPO` = `/leonardo_work/EUHPC_D30_031/alembic/alembic` on Leonardo.

| Seed | Final adapter (+ tokenizer) | Per-step adapters | Resumable checkpoints (newest 2) |
|---|---|---|---|
| 42 | `runs/verified4775_qdora/seed_42/adapter/` | `runs/verified4775_qdora/seed_42/adapters/step_{800,897}/adapter/` (copied by hand from the checkpoints) | `runs/verified4775_qdora/seed_42/checkpoints/checkpoint-{800,897}/` |
| 1 | `runs/verified4775_qdora_v2/seed_1/adapter/` | `runs/verified4775_qdora_v2/seed_1/adapters/step_{100,...,800,852}/adapter/` | `runs/verified4775_qdora_v2/seed_1/checkpoints/checkpoint-{800,852}/` |
| 2 | `runs/verified4775_qdora_v2/seed_2/adapter/` | `runs/verified4775_qdora_v2/seed_2/adapters/step_<N>/adapter/` | `runs/verified4775_qdora_v2/seed_2/checkpoints/` |

The final adapter and the last per-step adapter (step 897 / 852) hold the same weights.

### Training loss

![Batch 1 train and validation loss](img/batch1_loss.png)

Train loss drops at each epoch boundary (the model sees the same records again), while validation
loss stays within 1.17-1.20 the whole run. Seed 42 has no validation curve (no split). Seed 2 to be
added when it finishes.

Per-step train loss is one batch and noisy; the average is over the whole run.

| Seed | Step 1 | Avg train loss | Token accuracy (start → end) |
|---|---|---|---|
| 42 | 2.11 | 1.009 | 0.63 → ~0.72 |
| 1 | 1.94 | 1.009 | 0.66 → ~0.71 |

Seed 1 validation loss (245 held-out records, every 100 steps):

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 852 |
|---|---|---|---|---|---|---|---|---|---|
| Val loss | 1.195 | 1.175 | 1.173 | 1.175 | **1.172** | 1.186 | 1.190 | 1.189 | 1.190 |
| Val token acc | 0.704 | 0.706 | 0.707 | 0.708 | 0.707 | 0.707 | 0.706 | 0.707 | 0.707 |

Validation loss bottoms out around step 300-500 (end of epoch 1 to mid epoch 2) and rises in epoch 3:
the third epoch likely overfits.

### Final adapter, config eval set

`data/eval/eval_all_v2.jsonl` (66 normal) + `data/eval/eval_tools.jsonl` (16 tool), judge
`openai/gpt-oss-20b`, protocol `paired-orders-v2-invalid-incomplete`, 132/132 verdicts valid.

| Metric | Base | Seed 42 | Seed 1 | Seed 2 |
|---|---|---|---|---|
| Win rate vs base (0.5 = tie) | – | **0.341** | **0.307** | pending |
| Check pass rate | 0.955 | 0.864 | 0.864 | |
| Bad-flag rate | 0.025 | 0.255 | 0.324 | |
| Answers with bad flags | 2 | 9 | 8 | |
| Tool decision accuracy | 1.0 | 0.875 | 0.875 | |
| Tool false-call rate | 0.167 | 0.5 | 0.5 | |
| Tool valid / AST / exec ok | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | |
| Tool grounded rate | 1.0 | 0.923 | 0.923 | |

Jobs: seed 42 `evaluate` 59917194 (its trailing `track` fails: no MLflow run; results are complete),
seed 1 59914593, seed 2 59915059.

**The final student is worse than the base model** on both seeds: it loses the judged comparison,
uses ~10x more invalid command-line flags, and calls tools on half the questions where none is needed.

### Per-step dev eval (dataset_v1 development slice)

106 normal + 23 tool items (`development_normal.jsonl`, `development_tools.jsonl`), same judge and
protocol, 212/212 verdicts valid. Base and student both answer at every step.

**Seed 42** (job 59914860; its trailing `track` fails: no MLflow run; results are complete):

| Metric | Base | Step 800 | Step 897 (final) |
|---|---|---|---|
| Win rate vs base (0.5 = tie) | – | **0.330** | **0.349** |
| Check pass rate | 0.943 | 0.915 | 0.915 |
| Bad-flag rate | 0.029 | 0.211 | 0.250 |
| Answers with bad flags | 3 | 9 | 9 |
| Tool decision accuracy | 0.957 | 0.913 | 0.913 |
| Tool false-call rate | 0.111 | 0.222 | 0.222 |
| Tool first-response pass rate | 0.957 | 0.870 | 0.870 |
| Tool raw call rate | 0.652 | 0.696 | 0.696 |
| Tool valid / AST / exec ok | 1.0 / 1.0 / 1.0 | 1.0 / 0.929 / 1.0 | 1.0 / 0.929 / 1.0 |
| Tool grounded rate | 1.0 | 1.0 | 1.0 |

Same picture as the config eval set: the student loses about 2:1 to the base and adds invalid flags.
Steps 800 and 897 barely differ.

**Seed 1** (job 59916409, steps 100-852) and **seed 2** (job 59915060): pending. Each step takes
~10-12 min to answer (base is re-answered every step), so a 9-step seed takes ~2.5 h.

| Step | Seed 42 win rate | Seed 1 win rate | Seed 2 win rate |
|---|---|---|---|
| 800 | 0.330 | | |
| 897 / 852 (final) | 0.349 | | |

### Findings: training data does not match the eval

Checked on `verified.jsonl` with the same `check_answer` the eval uses:

| | Training data (4,775 records) | Eval sets |
|---|---|---|
| Topics | ROCm systems/libraries/examples ~1,290, JAX 447, PyTorch + tutorials 569, Spark 393, Ray 350, Dask 267, Spack 182, Kokkos 103, ... | Slurm job arrays / requeue (config set 66/66, dev 66/106), Aalto Triton cluster usage (dev 40/106) |
| Records mentioning Slurm | **36 (0.8%)** | almost all |
| Records with a tool call | **3** (10 offer tools) | 16 (config) / 23 (dev) tool items |
| Answers with invalid Slurm flags | 0 | student 21-32% bad-flag rate |
| Task mix | concept 2,341, script 1,767, debug 394, howto 273 | |

The student is trained on GPU/Python-library material and tested on Slurm and tool use. It does
not learn Slurm from the teacher (the teacher data has no bad flags); it drifts from what the base
already knew, and invents flags and tool calls. More epochs make the drift worse (validation loss
rises in epoch 3).

### Next steps

1. **Data (the real fix):** generate teacher records from Slurm and cluster-usage documents (still
   excluding the held-out `slurm_job_arrays`, `slurm_requeue_signals` and Aalto docs) and a few
   hundred tool records with balanced call / no-call / ask outcomes; aim for Slurm + tools at
   30-50% of the mix.
2. **Train less:** `epochs: 1` (validation loss is lowest at steps 300-500). Confirm with seed 1's
   per-step dev win rates before changing.
3. **Gentler updates:** `learning_rate: 5e-5` (from 1e-4), optionally `lora_r: 8`. Limits drift,
   but only data can make the student better than base on Slurm.
4. **In-domain eval:** a small held-out slice on the training topics (ROCm, JAX, PyTorch) to check
   whether the student improves where it was trained.

Proposed batch 2: same data, `epochs: 1`, `learning_rate: 5e-5`, 3 seeds (~20 min each), to measure
how much of the regression is overtraining, while the data fix is prepared.

### Open questions

- Training data was checked by the verifier and gpt-oss-20b only; no human audit yet.

### Infrastructure notes

- The first evaluate job (59914592) failed in vLLM startup on a CUDA 13 `torchcodec` wheel; fixed in
  the vLLM venv with the CPU build and resubmitted as 59917194.
- Batch 1 stopped early (2026-10-10 ~19:45): seed 1 per-step dev eval (59916409, steps 100 and 200
  answered, not judged) and seed 2's evals (59915059, 59915060) were cancelled once the data/eval
  mismatch was found. Seed 2's adapters are kept in `runs/verified4775_qdora_v2/seed_2/` (48.6 min).

## Batch 2: 1 epoch, lr 5e-5, same data (2026-10-10)

Tests how much of batch 1's regression is overtraining / drift, with no data change.

- **Config:** `configs/qwen3_4b_qdora_b2.yaml` = batch 1 config with `epochs: 1` (was 3) and
  `learning_rate: 5.0e-5` (was 1e-4). Everything else unchanged (validation split 5%, save_steps 100,
  MLflow on).
- **Data:** same `verified.jsonl` (`3f99eefb…`), copied to `runs/b2_ep1_lr5e-5/`.
- **Code:** dev 3738b9b.
- **Jobs:** train 59918667 (array 0-2 = seeds 42, 1, 2), `evaluate` 59918668 and
  `evaluate_checkpoints` 59918671, each seed's evals starting when its own training finishes
  (`--dependency=aftercorr`).
- **Adapters:** `runs/b2_ep1_lr5e-5/seed_<n>/adapter/` (final), `.../adapters/step_<N>/adapter/`.
- **Training:** 284 steps, ~18.7 min per seed. Validation loss (step 100 → 200 → 284): seed 42
  1.178 → 1.166 → 1.165; seed 1 1.207 → 1.195 → 1.194; seed 2 1.175 → 1.160 → 1.158. Validation
  documents differ by seed, so compare a seed only with itself (seed 1 batch 1 best: 1.172).

### Batch 2 final adapter, config eval set

Jobs 59918668_{0,1,2}; `track` logged the results into each seed's MLflow run (first use of PR #29).

**Normal questions (66, judged + checked):**

| Metric | Base | Seed 42 | Seed 1 | Seed 2 | Batch 2 mean | Batch 1 mean |
|---|---|---|---|---|---|---|
| Win rate vs base | – | 0.367 | 0.292 | 0.318 | **0.326** | 0.324 |
| Check pass rate | 0.955 | 0.909 | 0.879 | 0.909 | 0.899 | 0.864 |
| Bad-flag rate | 0.025 | 0.216 | 0.244 | 0.200 | 0.220 | 0.290 |
| Answers with bad flags | 2 | 6 | 8 | 6 | 6.7 | 8.5 |

**Tool questions (16):**

| Metric | Base | Seed 42 | Seed 1 | Seed 2 | Batch 2 mean | Batch 1 mean |
|---|---|---|---|---|---|---|
| Decision accuracy | 1.0 | 0.875 | 0.812 | 0.875 | 0.854 | 0.875 |
| False-call rate | 0.167 | 0.333 | 0.500 | 0.333 | 0.389 | 0.500 |
| Correct call (AST) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Grounded rate | 1.0 | 0.833 | 0.846 | 0.833 | 0.837 | 0.923 |

**Verdict:** 1 epoch at half the learning rate does not fix it. The judged win rate is unchanged
(0.326 vs 0.324); invalid flags and false tool calls drop a little but stay far above the base.
The regression comes from the data, not from overtraining; settings alone will not get the student
past the base.

## Data: Slurm + tool teacher records (2026-10-10, in progress)

Fills batch 1's gap (0.8% Slurm, 3 tool traces) with the repo's own pipeline (`generate` → `verify`).

- **Config:** `configs/gen_slurm_tools.yaml` = batch 1 config with `questions_per_chunk: 30` (was 12),
  `tool_fraction: 0.5` (was 0.3), `tool_mix: {call: 0.4, ask: 0.2, none: 0.4}` (was 0.6/0.2/0.2: more
  no-call traces, since the students over-call tools).
- **Sources:** the 4 trainable Slurm seeds in `data/seeds/` (`slurm_sbatch_basics`, `slurm_srun_steps`,
  `slurm_gpu_jobs`, `slurm_monitoring`). `slurm_job_arrays` and `slurm_requeue_signals` stay held out
  (`seeds.eval_docs`); the Aalto dev docs are not in the seeds.
- **Teacher:** `Qwen/Qwen3-32B-AWQ` on one A100 (`slurm/generate.sbatch`), then `distillkit verify`.
- **Run 1** (job 59918760, `runs/gen_slurm_tools/`, 9.6 min incl. 8.3 min vLLM load): only **80
  questions** planned (45 tool-mode). The planner takes at most personas x task_types = 4 x 4 = 16
  questions per chunk and the 4 docs are 5 chunks, so `questions_per_chunk: 30` had no effect.
  `verify` kept **61/80**; rejected: tool_decision 7, tool_ungrounded_id 11, empty_or_too_long 1.
- **Run 2** (job 59919372, `runs/gen_slurm_tools_p12/`): 8 more personas (12 total) and
  `questions_per_chunk: 48` → **240 questions** planned: 122 prose, 52 call, 46 no-call, 20 ask;
  sbatch_basics 96, gpu_jobs 48, monitoring 48, srun_steps 48. (The config file now has run 2's
  settings; run 1 = same file with the original 4 personas.)
- **Plan:** batch 3 = 4,775 records + the verified Slurm/tool records, batch 2's settings.

Run 2 result (job 59919372, 8.5 min incl. 6.2 min vLLM load): `verify` kept **182/240**; rejected:
tool_ungrounded_id 30, tool_decision 20, tool_ungrounded_partition 2, near_duplicate 2,
bash_syntax 2, tool_call_invalid 2. Kept: 119 prose, 34 no-call, 17 call, 12 ask (24 multi-turn);
sbatch_basics 76, gpu_jobs 37, srun_steps 35, monitoring 34. 0 of 410 Slurm flags invalid; 0 exact
question overlap with any eval/dev file. Spot check: some answers reason weakly (e.g. "more CPU cores
per GPU avoids GPU out-of-memory"); automated gates only, no human review. Run 1's 61 records are not
used (same seeds, older 4-persona plan).

## Batch 3: + Slurm data, prose only (3a) vs prose + tools (3b) (2026-10-10)

Same settings as batch 2 (`configs/qwen3_4b_qdora_b2.yaml`: 1 epoch, lr 5e-5), new data. 3a vs 3b
separates the effect of Slurm prose from the effect of tool traces.

| | 3a: `runs/b3a_prose` | 3b: `runs/b3b_tools` |
|---|---|---|
| Records | 4,775 + 119 Slurm prose x3 = 5,132 | 4,775 + 182 Slurm (prose + tool) x3 = 5,321 |
| New-data share | 7.0% | 10.3% |
| `verified.jsonl` SHA-256 | `499fd901…` | `0124a446…` |
| Train / evaluate / eval-ckpt jobs | 59920006 / 59920007 / 59920008 | 59920009 / 59920014 / 59920016 |

- **Upsampling:** each new record appears 3 times (copies get `#up1`, `#up2` id suffixes) so the new
  data is not lost in 4,775 records; in one epoch that is 3 passes over it.
- **Validation split:** whole documents; checked for seeds 42, 1, 2 that no Slurm seed doc lands in
  validation (all new records are trained on).
- **Adapters:** `runs/b3{a_prose,b_tools}/seed_<n>/adapter/` and `.../adapters/step_<N>/adapter/`.
- **Data location (not in git, on purpose: size):** only on Leonardo under `$REPO`:
  new records `runs/gen_slurm_tools_p12/verified.jsonl` (182; rejected, questions and teacher
  logprobs in the same folder); combined files `runs/b3a_prose/verified.jsonl` and
  `runs/b3b_tools/verified.jsonl` (hashes above). To rebuild: base 4,775 + the new records (3a: prose
  only) appended 3 times, copies 2 and 3 with ids suffixed `#up1` / `#up2`.

Results: pending.

## Batch 3b-r8: rank 8 on the 3b data (2026-10-10)

Does a smaller adapter drift less from the base? Identical to 3b except the adapter size.

- **Config:** `configs/qwen3_4b_qdora_r8.yaml` = batch 2 config with `lora_r: 8` (was 16) and
  `lora_alpha: 16` (was 32; alpha kept at 2x rank, as in every batch).
- **Data:** the 3b file (`0124a446…`, 5,321 records), copied to `runs/b3b_r8/`.
- **Jobs:** train 59920258 (seeds 42, 1, 2), `evaluate` 59920260, `evaluate_checkpoints` 59920263.
- **Adapters:** `runs/b3b_r8/seed_<n>/adapter/` and `.../adapters/step_<N>/adapter/`.

Results: pending.

## Data prototype: When2Call "ask a follow-up" records (2026-10-10, not trained yet)

Public data search (Hugging Face, Kaggle, papers): no public Slurm Q&A dataset exists. The one
ready-made set aimed at our tool failure is [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call)
(CC-BY-4.0): when to call a tool, ask, or abstain. General function-calling sets (xLAM-60k, Glaive,
Hermes) are avoided: training on xLAM-60k is reported to make small models call tools *more* when no
tool fits ([Hammer, arXiv 2410.04587](https://arxiv.org/pdf/2410.04587)). `hpcgroup/hpc-instruct`
(MIT, 122k) is HPC *code* (MPI/CUDA/OpenMP), not Slurm usage: same mismatch as the 4,775 records.

- **What the SFT split is:** 15,000 two-message rows, none with a tool call: ~7,100 clarifying
  questions ("Could you provide the ticker IDs?") and ~7,900 refusals ("I'm unable to provide
  real-time information").
- **Kept:** only clarifying questions on rows offering >= 1 tool (7,109 candidates). Refusals are
  left out: our "none" mode means *answer from knowledge*, and refusal data could teach the student
  to decline Slurm questions it should answer.
- **Conversion:** xLAM-style tool schemas (`dict`/`str`/`int`, `required` outside `parameters`)
  rewritten as JSON Schema like our own tools; each row its own `doc_id` (`when2call:<row>`), with
  `source_origin` and `license` fields. A 300-row random sample (seed 0) renders through
  `records.training_example`; 1-5 tools per row.
- **Spot check (12 rows):** 11 ask for a genuinely missing required argument; 1 is doubtful
  (asks how many items to skip). Off-domain: 3 of 300 mention jobs/GPUs/clusters at all.
- **Where:** converter and sample only in the session scratchpad so far (not on Leonardo, not in git).
- **Proposed use:** batch 4 = 3b data + these 300 records, to test whether general "ask instead of
  call" examples lower the false-call rate without hurting normal answers.
