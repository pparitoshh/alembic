# Topic pipeline: a reusable sources → Q&A → review → clean-data utility

**Goal.** Turn the pipeline that produced the HPC10K data (docs/plan/HPC10K_DATA.md) into one modular tool
that can build a quality-gated distillation dataset for *any* documentation topic (e.g. PyTorch distributed,
Kubernetes, PostgreSQL) by writing a **topic pack**, not code. One command: `distillkit build --topic <pack>
--target 10000` → `runs/<topic>/clean.jsonl` plus a report.

**Constraint.** HPC is topic #1 and must keep working bit-for-bit: frozen policy/protocol versions, the
dataset_v1 manifest hash (`config.py` defaults) and archived replay tests stay unchanged. New behaviour is
opt-in via new versioned ids; the HPC pack is the existing settings moved, with a replay test proving equality.

## 1. What works today (keep)

| Stage | Module | Status after HPC10K |
|---|---|---|
| fetch sources | `fetch_sources.py` | pinned URLs, licence, SHA-256, held-out block filter |
| seeds / chunking | `seeds.py`, `source_registry.py`, `source_resolver.py` | chunk_chars, eval docs held out before generation |
| questions + answers | `generate.py`, `teacher.py` | Qwen3-32B-AWQ, personas × task types, tool modes, resumable |
| review | `review.py`, `support_v3.py` | Gemma 4 source-support-v3; malformed reviews redone on rerun; whitespace fix → 97% valid |
| verify | `verify.py`, `checks.py`, `toolcheck.py` | deterministic gates + review verdicts |
| decontam | `heldout.py` | against eval files |
| sharding | `slurm/submit_gated_shards.sh`, `generate_gated.sbatch` | N GPUs in parallel, resume on resubmit |

## 2. What is HPC-specific today (to move into a pack)

- `generate.py`: `Q_SYSTEM`, `A_SYSTEM`, mode prompts ("users of an HPC cluster", "Slurm cluster", partitions).
- `checks.py`: Slurm flag validation (`SLURM_CMDS`, `data/slurm_flags.txt`).
- `tools.py`, `job_status_guard.py`, `scenario_plan.py`: the mock cluster and its tool contract.
- `config.py`: `tool_application_policy` literals `hpc-tools-*`; `student.system_prompt`.
- `data/sources/hpc10k_sources.json` (incl. `heldout_pattern`), `data/gold/*`, personas in `configs/gen_hpc10k.yaml`.

## 3. Design

### 3.1 Topic pack (the only thing a new topic writes)

```
topics/<name>/
  topic.yaml          # id, version, description, student system prompt, personas, task_types, targets
  sources.json        # pinned URLs + licence + format; heldout_pattern; eval_docs
  prompts/            # question.j2, answer.j2, (mode_call.j2, mode_ask.j2 if tools) — versioned
  gold/               # 2–3 few-shot anchors (prose.json, tool_trace.json)
  eval/               # held-out eval questions (written BEFORE generation)
  checks.py           # optional: domain checks registered by name (e.g. slurm_flags)
  tools/              # optional: tool schemas + mock world + guard; tool_fraction 0 = prose only
```

`topics/hpc/` is built first by moving today's settings; `topic.version` + prompt file hashes go into every
record (`prompt_version` already exists) and into the run manifest.

### 3.2 Core stages (topic-agnostic, each `run(cfg) -> Path`, resumable, idempotent)

`fetch → chunk → plan → questions → answers → review → verify → decontam → merge → report`

- **plan** (new, small): from chunks × personas × task types × target size, compute `questions_per_chunk`
  so expected clean ≈ target / measured keep rate (73% now); writes `plan.json`. Replaces hand-tuned top-ups.
- **merge** (new): union of shard `clean.jsonl`, exact-id + normalized-question + near-duplicate (MinHash,
  `dedup_threshold`) dedupe, gold-doc duplicates kept once; writes `clean.jsonl` + `merge_report.json`.
- **report** (new): per-stage, per-shard, per-doc, per-persona counts; pending/reject reasons; malformed rate;
  `finish_reason` histogram; 100-record spot-check sample. (What we computed by hand in HPC10K_DATA.md.)
- Checks become a registry: `verify.checks: [length, dedup, slurm_flags]`; the pack names which apply.
- Tools become optional: with no `tools/`, tool modes are disabled and the mock replay gate is skipped.

### 3.3 Model serving profiles (learned the hard way)

One `serving.yaml` profile per role, shared by all topics:
- teacher: Qwen3-32B-AWQ, `--max-model-len 16384`.
- reviewer: Gemma 4 26B-A4B, `--max-model-len 24576 --structured-outputs-config
  {"backend":"xgrammar","disable_any_whitespace":true}` (backend `auto` rejects that flag; without it ~47% of
  reviews ran whitespace to `max_tokens`).
- judge (eval): gpt-oss-20b; cross-judge Gemma.
Log `finish_reason`/`stop_reason` for every request (done for review in `3c1b672`; extend to generate).

### 3.4 Orchestration

`distillkit build --topic topics/<name> --target N [--shards K] [--dry-run]`:
1. fetch + chunk on the login node; hold out eval docs; write manifest.
2. plan; split into K shards (largest-doc-first, as `submit_gated_shards.sh`).
3. submit one job per shard (generate → review → verify → decontam); **back up shard outputs before any
   rerun** (as `runs/hpc10k_backup_20261011`).
4. watch: poll every 25 min; on finish, if malformed > 2% or failed reviews, resubmit review-only.
5. merge + report; if clean < target, plan a top-up with new personas/question seeds (new ids only) and repeat.
Cluster submission stays behind an explicit confirmation flag (shared cluster).

## 4. Phases and acceptance

| Phase | Work | Done when |
|---|---|---|
| P0 | Freeze: tag current HPC pipeline; record HPC10K final counts | tag pushed, counts in HPC10K_DATA.md |
| P1 | `topics/hpc/` pack; prompts → templates; checks registry; tools optional | full test suite green; replay test: same prompts/hashes for HPC as today; dataset_v1 hash unchanged |
| P2 | `plan`, `merge`, `report` stages | HPC10K shards merged by `merge`; report matches the hand counts |
| P3 | serving profiles + `build` orchestrator (dry-run first) | `build --topic topics/hpc --dry-run` reproduces today's 5 shard commands |
| P4 | Second topic pilot (prose-only, e.g. PyTorch distributed docs, ~30 docs, 500 clean) | pilot keep rate ≥ 60%, malformed ≤ 3%, 0 eval overlap, spot-check 50 records |
| P5 | Docs: "add a topic in 1 hour" guide + pack template + `distillkit new-topic` scaffold | someone else builds a pack from the guide alone |

## 5. Quality gates (every topic, unchanged from HPC)

- Eval docs and held-out patterns removed **before** generation; decontam after.
- Independent reviewer ≠ teacher; verify requires a parsed review for every kept record.
- Deterministic checks first, LLM judgement second; pending is never silently kept.
- Every record carries topic id/version, prompt version, teacher, reviewer, source doc + chunk id.
- Domain mix sanity: report share of records actually on the topic (lesson from the 14,304 set: 35/14,304
  mentioned Slurm) and of tool traces vs. the plan.

## 6. Open questions

- Pack location: `topics/` in this repo vs. a separate data repo.
- Re-use `support_v3` unchanged for all topics, or a v4 with topic-provided reviewer guidance (new id).
- Mock worlds for tool topics other than Slurm: per-pack Python or a declarative JSON world.
