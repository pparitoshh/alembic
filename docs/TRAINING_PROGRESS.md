# Training progress

Log of every student batch: data, config, jobs, results. Numbers as reported by the jobs, not rounded
guesses. Student `Qwen/Qwen3-4B-Instruct-2507`, QDoRA (`method: dora`, 4-bit), 1 x A100 per seed
(Booster, `boost_usr_prod`). Paths are relative to `$REPO` = `/leonardo_work/EUHPC_D30_031/alembic/alembic`.

| Result | File | Written by |
|---|---|---|
| Train / val loss per step | `seed_<n>/adapters/step_<N>/adapter/trainer_state.json`, MLflow | `slurm/train.sbatch` |
| Final adapter, config eval set | `seed_<n>/eval_summary.json` | `slurm/evaluate.sbatch` |
| Every kept step, dataset_v1 dev slice | `seed_<n>/adapters/step_<N>/eval_summary.json`, MLflow `dev/...` | `slurm/evaluate_checkpoints.sbatch` |

MLflow: `$REPO/mlruns`, experiment `distillkit`, one run per seed named after its run dir (`uvx mlflow ui`).
Adapters: `<run>/seed_<n>/adapter/` (final, + tokenizer), `<run>/seed_<n>/adapters/step_<N>/adapter/` (per step).

## Summary: final adapters, config eval set

`data/eval/eval_all_v2.jsonl` (66 normal) + `data/eval/eval_tools.jsonl` (16 tool), judge
`openai/gpt-oss-20b`, protocol `paired-orders-v2-invalid-incomplete`. Mean over seeds 42, 1, 2
(B1: 42, 1), seed range in brackets. Correct-call (AST) accuracy is 1.0 for every batch and the base.

| Batch | Change | Win rate vs base | Check pass | Bad-flag rate | Tool decision | False-call | Grounded |
|---|---|---|---|---|---|---|---|
| Base | – | – | 0.955 | 0.025 | 1.0 | 0.167 | 1.0 |
| B1 | 4,775 records, 3 ep, lr 1e-4 | 0.324 [0.307-0.341] | 0.864 | 0.289 | 0.875 | 0.500 | 0.923 |
| B2 | 1 ep, lr 5e-5 | 0.326 [0.292-0.367] | 0.899 | 0.220 | 0.854 | 0.389 | 0.838 |
| B3a | + Slurm prose (119 x3) | **0.359** [0.333-0.379] | 0.904 | 0.098 | 0.875 | 0.333 | 0.833 |
| **B3b** | **+ Slurm prose and tools (182 x3)** | **0.359** [0.356-0.364] | **0.924** | **0.059** | **1.000** | **0.000** | **1.000** |
| B3b-r8 | 3b, rank 8 | 0.355 [0.326-0.375] | 0.919 | 0.082 | 0.979 | 0.167 | 0.970 |
| B5 | HPC10K snapshot 7,743 + 2k retained | pending | pending | pending | pending | pending | pending |

**Conclusions (B1-B3)**
1. Settings alone (B1 → B2) change nothing: the regression comes from the data, not overtraining.
2. On-topic Slurm prose (B3a) halves invalid flags (0.22 → 0.10) and lifts the win rate to 0.36.
3. Slurm tool traces (B3b) are the best run: flags 0.06; on tool questions it matches or beats the base
   (decision 1.0, 0 false calls vs base 0.167, no invented job ids).
4. Rank 8 is no better than 16 (more flags, false calls back to 0.167): keep rank 16.
5. Earlier checkpoints do not help: dev win rate flat at 0.30-0.35 across steps.
6. Still below the base on normal questions (0.36 vs 0.5) with 182 new records (3% of the mix) → HPC10K
   data (`docs/plan/HPC10K_DATA.md`, branch `feat/data-hpc10k`), batch 5.

**Per-step dev slice** (dataset_v1 development, 106 normal + 23 tool), win rate vs base:

| Run | Step 100 | Step 200 | Final step |
|---|---|---|---|
| B1 seed 42 | – | – | 0.330 (800), 0.349 (897) |
| B2 seed 42 | 0.325 | 0.316 | 0.314 (284) |
| B2 seed 1 | 0.333 | 0.307 | 0.300 (284) |
| B2 seed 2 | 0.302 | 0.300 | pending |

Dev-slice false-call rate 0.222 at every B2 step (base 0.111). B3 per-step dev: pending (jobs 59920008,
59920016, 59920263).

## Batch 1: verified_qwen_4775 (2026-10-10)

- **Data:** `data/teacher/verified_qwen_4775/verified.jsonl` (PR #25), 4,775 records, SHA-256 `3f99eefb…`;
  tokens/example median 164, max 1,520, 0 over `max_length`.
- **Config** `configs/qwen3_4b_qdora.yaml`: r/alpha/dropout 16/32/0.0; lr 1e-4 cosine; 3 epochs; batch 4 x
  grad accum 4 (eff. 16); max_length 2048; bf16; save_steps 100 (newest 2 kept); val_fraction 0.05
  (seeds 1, 2 only). ~3.1 s/step.

| Seed | Run dir | Code | Val split | MLflow | Train job | Steps | Wall | Eval job |
|---|---|---|---|---|---|---|---|---|
| 42 | `runs/verified4775_qdora` | 5873df7 (PR #27) | none | no | 59911612 | 897 | 51.5 min | 59917194 |
| 1 | `runs/verified4775_qdora_v2` | a15dc23 (PR #28) | 4,530 / 245 | yes | 59912914 | 852 | 47.8 min | 59914593 |
| 2 | `runs/verified4775_qdora_v2` | 3738b9b (PR #29) | 4,530 / 245 | yes | 59915034 | 852 | 48.6 min | 59915059 (cancelled) |

Seed 42 (older code) has only steps 800, 897 as per-step adapters (copied by hand from
`checkpoints/checkpoint-{800,897}`); seed 1 has steps 100-852. The final adapter = the last step's weights.

**Loss** (![Batch 1 loss](img/batch1_loss.png)): step-1 loss 2.11 / 1.94, avg 1.009 / 1.009, token acc
0.63 → ~0.72 / 0.66 → ~0.71 (seed 42 / 1). Train loss drops at each epoch boundary; seed 1 val loss:

| Step | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 852 |
|---|---|---|---|---|---|---|---|---|---|
| Val loss | 1.195 | 1.175 | 1.173 | 1.175 | **1.172** | 1.186 | 1.190 | 1.189 | 1.190 |
| Val token acc | 0.704 | 0.706 | 0.707 | 0.708 | 0.707 | 0.707 | 0.706 | 0.707 | 0.707 |

Lowest at steps 300-500; epoch 3 likely overfits.

**Per seed, config eval set (132/132 verdicts valid):**

| Metric | Base | Seed 42 | Seed 1 |
|---|---|---|---|
| Win rate | – | 0.341 | 0.307 |
| Check pass / bad-flag rate | 0.955 / 0.025 | 0.864 / 0.255 | 0.864 / 0.324 |
| Answers with bad flags | 2 | 9 | 8 |
| Tool decision / false-call / grounded | 1.0 / 0.167 / 1.0 | 0.875 / 0.5 / 0.923 | 0.875 / 0.5 / 0.923 |
| Tool valid / AST / exec ok | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |

**Seed 42 dev slice** (job 59914860, 212/212 valid; `track` fails without MLflow, results complete):

| Metric | Base | Step 800 | Step 897 |
|---|---|---|---|
| Win rate | – | 0.330 | 0.349 |
| Check pass / bad-flag rate | 0.943 / 0.029 | 0.915 / 0.211 | 0.915 / 0.250 |
| Answers with bad flags | 3 | 9 | 9 |
| Tool decision / false-call / first-response pass | 0.957 / 0.111 / 0.957 | 0.913 / 0.222 / 0.870 | 0.913 / 0.222 / 0.870 |
| Tool raw call rate | 0.652 | 0.696 | 0.696 |
| Tool valid / AST / exec ok / grounded | 1.0 / 1.0 / 1.0 / 1.0 | 1.0 / 0.929 / 1.0 / 1.0 | 1.0 / 0.929 / 1.0 / 1.0 |

Each dev step takes ~10-12 min (base re-answered every step).

**Finding: the data does not match the eval** (same `check_answer` as the eval):

| | Training data (4,775) | Eval sets |
|---|---|---|
| Topics | ROCm ~1,290, PyTorch + tutorials 569, JAX 447, Spark 393, Ray 350, Dask 267, Spack 182, Kokkos 103, ... | Slurm arrays / requeue (config 66/66, dev 66/106), Aalto Triton (dev 40/106) |
| Mention Slurm | **36 (0.8%)** | almost all |
| With a tool call | **3** (10 offer tools) | 16 (config) / 23 (dev) |
| Invalid Slurm flags | 0 | student 21-32% |
| Task mix | concept 2,341, script 1,767, debug 394, howto 273 | |

The student drifts from what the base knew and invents flags and tool calls; more epochs make it worse.
Next steps then: Slurm + tool data (30-50% of the mix), 1 epoch, lr 5e-5 (optionally r 8), an
in-domain eval slice. Open: no human audit of the training data (verifier + gpt-oss-20b only).

**Infra:** eval 59914592 failed in vLLM startup on a CUDA 13 `torchcodec` wheel → CPU build in the vLLM
venv, resubmitted as 59917194. Stopped early 2026-10-10 ~19:45: seed 1 dev eval 59916409 (steps 100, 200
answered, not judged) and seed 2 evals 59915059, 59915060 cancelled once the mismatch was found.

## Batch 2: 1 epoch, lr 5e-5, same data (2026-10-10)

Is B1's regression overtraining? `configs/qwen3_4b_qdora_b2.yaml` = B1 config with `epochs: 1`,
`learning_rate: 5.0e-5`. Same data (`3f99eefb…`) in `runs/b2_ep1_lr5e-5/`, code 3738b9b. Jobs: train
59918667 (array 0-2), `evaluate` 59918668, `evaluate_checkpoints` 59918671 (`--dependency=aftercorr`);
`track` logged into MLflow (first use of PR #29). 284 steps, ~18.7 min/seed. Val loss (steps 100 → 200 →
284): seed 42 1.178 → 1.166 → 1.165; seed 1 1.207 → 1.195 → 1.194; seed 2 1.175 → 1.160 → 1.158 (val
docs differ by seed; seed 1's B1 best was 1.172).

| Metric | Base | Seed 42 | Seed 1 | Seed 2 | B2 mean | B1 mean |
|---|---|---|---|---|---|---|
| Win rate | – | 0.367 | 0.292 | 0.318 | **0.326** | 0.324 |
| Check pass rate | 0.955 | 0.909 | 0.879 | 0.909 | 0.899 | 0.864 |
| Bad-flag rate | 0.025 | 0.216 | 0.244 | 0.200 | 0.220 | 0.290 |
| Answers with bad flags | 2 | 6 | 8 | 6 | 6.7 | 8.5 |
| Tool decision | 1.0 | 0.875 | 0.812 | 0.875 | 0.854 | 0.875 |
| False-call rate | 0.167 | 0.333 | 0.500 | 0.333 | 0.389 | 0.500 |
| Grounded rate | 1.0 | 0.833 | 0.846 | 0.833 | 0.837 | 0.923 |

Verdict: unchanged win rate; flags and false calls drop a little. The problem is the data.

## Data: Slurm + tool teacher records (2026-10-10)

`configs/gen_slurm_tools.yaml` (`tool_fraction: 0.5`, `tool_mix` call/ask/none 0.4/0.2/0.4, was
0.6/0.2/0.2) on the 4 trainable Slurm seeds (`slurm_sbatch_basics`, `slurm_srun_steps`, `slurm_gpu_jobs`,
`slurm_monitoring`); `slurm_job_arrays`, `slurm_requeue_signals` held out, Aalto docs excluded. Teacher
`Qwen/Qwen3-32B-AWQ` (`slurm/generate.sbatch`), then `verify`.

| Run | Job, dir | Plan | Kept | Rejected |
|---|---|---|---|---|
| 1 | 59918760, `runs/gen_slurm_tools/` (9.6 min, 8.3 load) | 4 personas, `questions_per_chunk: 30` → 80 (45 tool); cap is personas x task types = 16/chunk, 5 chunks | 61/80 (not used) | tool_ungrounded_id 11, tool_decision 7, empty_or_too_long 1 |
| 2 | 59919372, `runs/gen_slurm_tools_p12/` (8.5 min, 6.2 load) | 12 personas, 48/chunk → 240: 122 prose, 52 call, 46 none, 20 ask; sbatch_basics 96, others 48 | **182/240** | tool_ungrounded_id 30, tool_decision 20, partition 2, near_dup 2, bash_syntax 2, call_invalid 2 |

Run 2 kept: 119 prose, 34 none, 17 call, 12 ask (24 multi-turn); sbatch_basics 76, gpu_jobs 37,
srun_steps 35, monitoring 34. 0 of 410 Slurm flags invalid; 0 exact overlap with eval/dev. Spot check:
some weak reasoning (e.g. "more CPU cores per GPU avoids GPU OOM"); automated gates only. The config file
holds run 2's settings.

## Batch 3: + Slurm prose (3a) vs prose + tools (3b), and 3b-r8 (2026-10-10)

B2 settings, new data; 3a vs 3b separates prose from tool traces; r8 = 3b with `lora_r: 8`, `lora_alpha:
16` (`configs/qwen3_4b_qdora_r8.yaml`).

| | 3a `runs/b3a_prose` | 3b `runs/b3b_tools` | 3b-r8 `runs/b3b_r8` |
|---|---|---|---|
| Records | 4,775 + 119 prose x3 = 5,132 | 4,775 + 182 x3 = 5,321 | as 3b |
| New-data share | 7.0% | 10.3% | 10.3% |
| `verified.jsonl` SHA-256 | `499fd901…` | `0124a446…` | `0124a446…` |
| Train / eval / eval-ckpt jobs | 59920006 / 59920007 / 59920008 | 59920009 / 59920014 / 59920016 | 59920258 / 59920260 / 59920263 |
| Steps, time per seed | 284+, ~18-21 min | 316, ~24.5 min | |

Upsampling: copies 2, 3 get ids `#up1`, `#up2`. Validation split by whole document; no Slurm seed doc
in validation for seeds 42, 1, 2. 3b seed 42 val loss 1.171 → 1.156 → 1.154 → 1.154 (steps 100, 200, 300,
316); 0 over `max_length` (median 168, max 1,690 tokens). Data only on Leonardo (size): new records
`runs/gen_slurm_tools_p12/verified.jsonl`; rebuild = base 4,775 + new records (3a: prose only) x3.

## Data prototype: When2Call "ask" records (2026-10-10, dropped)

No public Slurm Q&A dataset exists. [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call)
(CC-BY-4.0) targets call / ask / abstain; general function-calling sets (xLAM-60k, Glaive, Hermes) are
avoided (xLAM-60k makes small models over-call, [Hammer, arXiv 2410.04587](https://arxiv.org/pdf/2410.04587));
`hpcgroup/hpc-instruct` (MIT, 122k) is HPC code, same mismatch as the 4,775. SFT split: 15,000 two-message
rows, no tool calls: ~7,100 clarifying questions, ~7,900 refusals. Kept clarifying questions on rows with
>= 1 tool (7,109; refusals could teach declining Slurm questions); schemas converted to JSON Schema, own
`doc_id` `when2call:<row>`, `source_origin` + `license`; 300-row sample (seed 0) renders, 1-5 tools/row.
Spot check 12: 11 ask for a missing required argument, 1 doubtful; 3/300 mention jobs/GPUs/clusters.
Scratchpad only. **Not used:** 3b's own traces already reach 0 false calls.

## Data: HPC10K quality-gated Slurm / cluster-usage records (2026-10-11)

Plan and gate counts: `docs/plan/HPC10K_DATA.md` (branch `feat/data-hpc10k`). 34 pinned Slurm docs + HPC
Carpentry pages, 5 shards: Qwen3-32B-AWQ generates → Gemma 4 26B-A4B reviews each answer against its
source (source-support-v3) → deterministic verify → decontam against all held-out eval questions.

| Step | Result |
|---|---|
| First pass | 14,670 generated, 4,485 unique clean; ~47% of reviews malformed (whitespace after the JSON until `max_tokens`, `finish_reason=length`) |
| Fix | reviewer `--structured-outputs-config {"backend":"xgrammar","disable_any_whitespace":true}` + `--max-model-len 24576`; pilot of 126 (job 59972524): kept 62 → **92**, malformed 33 → **0** |
| Re-review s2-s5 (jobs 59974759-62) | clean 905/920/932/921 → **1,616/1,609/1,589/1,585**; 97% of redone reviews valid; 2 of 3,678 old clean dropped as near-duplicates |
| Top-up (`configs/gen_hpc10k_topup.yaml`: 4 new personas, 9 q/chunk) | t5 kept 806/1,431; t1-t4 and s1 re-review running; expected ~11k after dedupe |

Pre-re-review backup: `runs/hpc10k_backup_20261011/`.

## Batch 5: HPC10K snapshot + 2k retained (2026-10-11, submitted)

`runs/b5_hpc10k/verified.jsonl`, **9,743 records**: 7,743 HPC10K clean (s2-s5 re-reviewed, t5, s1 from
the backup; deduped by id and normalized question) + 2,000 random (seed 42) from verified_qwen_4775.
Modes: prose 7,997, none 1,061, ask 498, call 187. Slurm / cluster share ≈ 80% (B3b ~3%). B2 settings,
seeds 42, 1, 2; train 59980667 (array 0-2), eval 59980668 (`aftercorr`). Goal: beat B3b (results in the
summary table). Not final data: batch 6 on the full merged HPC10K set (~11k + retained) follows.
