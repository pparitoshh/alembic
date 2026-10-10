# Plan: ~10k quality-gated Slurm / cluster-usage training records

Status: running, 2026-10-10 (see **Status and handover** at the end). Branch `feat/data-hpc10k`. Results so far live in
`docs/TRAINING_PROGRESS.md` (branch `docs/training-progress`).

## Why

Batch 1 trained on 4,775 records that are 99% GPU/Python-library material (ROCm, JAX, PyTorch,
Spark, ...); only 36 mention Slurm and 3 contain a tool call. The eval is Slurm and tool use, and
the student lost to the base (win rate ~0.32, 22-32% invalid Slurm flags, 50% false tool calls).

Batch 3 added just 182 teacher records from 4 Slurm seed docs (x3 upsampled):

| Final adapter, config eval set | Base | Batch 2 | 3a (+Slurm prose) | 3b (+Slurm prose + tools) |
|---|---|---|---|---|
| Win rate vs base (normal) | – | 0.326 | 0.359 | 0.359 |
| Bad-flag rate | 0.025 | 0.220 | 0.098 | 0.059 |
| Tool false-call rate | 0.167 | 0.389 | 0.333 | **0.0** |
| Tool decision accuracy | 1.0 | 0.854 | 0.875 | **1.0** |

(Means over seeds 42, 1, 2.) On-topic data works, but 182 records are 3% of the mix and the win rate is still below 0.5. The
limit is source text: 4 seed docs (~5k characters) cap generation at ~250 questions.

## Target

~10k kept records: ~7-8k new Slurm / cluster-usage records (≈70% prose, 30% tool traces) plus
~2k of the existing 4,775 for retention.

| Step | Rate seen today | For 10k kept |
|---|---|---|
| Questions per 1,500-char chunk | 48 (12 personas x 4 task types) | ~280 chunks |
| Source text | 1,500 chars / chunk | ~400k chars, ~150-250 docs |
| `verify` keep rate | 76% (182/240) | ~13k generated |
| Teacher time (Qwen3-32B-AWQ, 1 A100) | ~2 min / 240 + 6-8 min load | ~2 h |

## Sources (pinned by URL, licence and content hash)

| Source | Licence | Use | Excluded |
|---|---|---|---|
| Slurm documentation, slurm.schedmd.com: man pages `sbatch`, `srun`, `salloc`, `squeue`, `sacct`, `scontrol`, `sinfo`, `scancel`, `sstat`, `sprio`, `sshare`, `sattach`, `sbcast`; guides quickstart, gres, mpi_guide, heterogeneous_jobs, cpu_management, mc_support, faq, job_exit_code | GPL-2.0 (Slurm source tree, `COPYING`) | scheduler knowledge | `job_array`, `preempt` pages; every block mentioning job arrays, requeue, signals or preemption (the eval's held-out topics) |
| HPC Carpentry *Introduction to HPC*, rendered lesson pages (carpentries-incubator/hpc-intro, commit `7081080`) | CC-BY-4.0 | general cluster use: connecting, scheduler basics, modules, environment, transfers, parallel jobs, resources | `21-array` (job arrays) |
| Not used | | | Aalto Scientific Computing guides (dataset_v1 dev slice); `data/seeds/slurm_job_arrays.md`, `slurm_requeue_signals.md` (config eval set) |

Leonardo's login node reaches both sites (checked 2026-10-10). More centre guides (e.g.
CINECA/Leonardo user guide) can be added after a licence check.

## Pipeline: one chained Slurm job, each gate must pass before the next

1. **fetch** (login node, network): download pages → markdown seed docs in
   `$REPO/runs/hpc10k/seeds/`, drop held-out blocks, write `sources_manifest.json` (URL, licence,
   fetched bytes SHA-256, seed SHA-256, dropped-block count, fetch time).
2. **generate** (GPU, teacher `Qwen/Qwen3-32B-AWQ`): existing `distillkit generate`, 12 personas,
   48 questions per chunk, tool fraction 0.3 with mix call 0.4 / ask 0.2 / none 0.4.
3. **review** (GPU, independent reviewer `google/gemma-4-26B-A4B-it`): new stage running the
   existing `source-support-v3` review on every generated record: each answer unit must be
   supported by the source chunk or replayed tool results. Resumable.
4. **verify** (CPU): existing gates (bad flags, bash syntax, tool replay and grounding, decision,
   near-duplicates) plus `require_grounding_review: true`: unsupported → rejected, uncertain →
   pending (not kept).
5. **decontam** (CPU): new stage; drops records whose question overlaps *any* held-out question in
   `data/eval/**` (config eval set, dataset_v1 development and final), not only `eval.file`.
6. **report**: counts per gate, per source, per mode; a 100-record sample for human spot-check.

Everything is written under `$REPO/runs/hpc10k/` on Leonardo; nothing goes into git (size).

## Phases

1. **Pilot (~1k):** first ~30 docs through the full chain. Check gate pass rates and the
   spot-check sample. Train batch 5 = 4,775 + pilot (batch 2 settings, 3 seeds); continue only if
   it beats 3b.
2. **Scale (~10k):** all sources. Batch 6 = new data + ~2k retained records.
3. Final-slice evaluation only once, on the setting chosen from dev results.

## Code changes (this branch, with hermetic tests)

- `src/distillkit/fetch_sources.py` + source list (`data/sources/hpc10k_sources.json`).
- `src/distillkit/review.py`: `distillkit review` stage (reviewer = `judge` endpoint).
- `src/distillkit/heldout.py`: `distillkit decontam` stage.
- `configs/gen_hpc10k.yaml`, `slurm/generate_gated.sbatch` (generate → review → verify → decontam).
- No `config.py` change (it would invalidate the dataset_v1 manifest hash).

Not used: nvidia/When2Call (prototyped; general-domain "ask" examples). 3b's own tool traces
already bring false calls to 0, so it is dropped (decision 2026-10-10).

## Open decisions

- Human spot-check of the 100-record sample: who (user / Vaibhav / second model only)?
- Extra sources beyond Slurm docs + HPC Carpentry.
- Retention share of the original 4,775 records (2k proposed).

## Status and handover (2026-10-10, night)

Leonardo checkout `$REPO` (= `/leonardo_work/EUHPC_D30_031/alembic/alembic`, shared with Vaibhav) is on
this branch at `d85d066`. Return it to `dev` once PR #31 is merged. All data is under `$REPO/runs/`, not in git.

**Fetch (done).** `runs/hpc10k/seeds/`: 34 docs, 1,017,342 chars, 181 held-out blocks dropped →
800 chunks x 18 questions (`questions_per_chunk: 18`) = 14,400 questions.

**Pilot 1 (job 59923908, `runs/hpc10k_pilot100`, `--limit` docs, 126 generated): kept only 3/126.**
- 114 reviews were malformed JSON: truncated at `max_tokens` 4096 (median review ~13k chars, one
  explanation per answer unit) or stuck in a greedy repetition loop ("- own- own ...").
- 6 tool traces: "tool evidence does not match complete ordered mock replay"; verify rejects those
  anyway (`tool_ungrounded_id`). 1 `tool_decision` reject.
- Fix (`d85d066`): reviewer `max_tokens` 8192 and `repetition_penalty` 1.05 (env `REVIEW_MAX_TOKENS`,
  `REVIEW_REPETITION_PENALTY` in `slurm/generate_gated.sbatch`); `review` now drops malformed reports and
  redoes them on rerun. Concurrency stays 16: Gemma's KV cache (4.95 GiB, ~55k tokens) is already ~100% full.
- Timing: teacher load ~6 min, reviewer load ~7 min, review 8-16 records/min at concurrency 16.

**Pilot 2 (job 59925542, same run dir, resumed; 20 min): kept 62/126** (was 3).
- `[verify] kept 62/126; rejected: {'tool_ungrounded_id': 6, 'unsupported_claim': 5, 'tool_decision': 1}; pending review: 52`;
  decontam dropped 0 (391 held-out questions, 11 files).
- Pending 52: 33 still "malformed review", 8 "supported unit lacks evidence", 4 "citation absent/altered",
  3 "unknown or duplicate target unit", 4 no reason.
- The 34 malformed reviews are *not* truncated now: 0.8-3k chars, all start `{"units": [{` and stop right
  after the units list (`... } ]`), without the required `uncertainty` field. Odd, since the request sends a
  `json_schema` response_format (vLLM guided decoding should forbid that). To check tomorrow: is guided
  decoding actually applied for Gemma 4 in our vLLM? Does `repetition_penalty` 1.05 cause the early stop
  (rerun those 34 without it)? A retry pass for malformed reports is the fallback.
- **Test submitted: job 59928215** (`dk-rp1-test`), a copy of the pilot in `runs/hpc10k_pilot100_rp1` with
  `REVIEW_REPETITION_PENALTY=1.0`; only the 34 malformed + 6 failed rows are re-reviewed. Read
  `grep -aE '^\[(review|verify)\]' slurm/logs/dk-rp1-test-59928215.out`: if malformed drops to ~0, set the
  default penalty to 1.0 in `slurm/generate_gated.sbatch` and re-review the shards.
- Not urgent for the shards: `review` redoes malformed reports on every rerun, so after a fix just rerun
  review + verify + decontam on each shard dir (no regeneration). At ~49% kept, 14,400 questions give
  ~7k records; fixing the malformed ~27% would add up to ~3.5k more.

**Full run: 5 shards submitted before pilot 2 finished (user's call)** with
`slurm/submit_gated_shards.sh` (24 h limit each):

| Shard | Job | Run dir | Docs | Chars |
|---|---|---|---|---|
| 1 | 59925723 | `runs/hpc10k_s1` | 6 | 203,800 |
| 2 | 59925724 | `runs/hpc10k_s2` | 7 | 203,818 |
| 3 | 59925725 | `runs/hpc10k_s3` | 7 | 203,183 |
| 4 | 59925726 | `runs/hpc10k_s4` | 7 | 203,259 |
| 5 | 59925727 | `runs/hpc10k_s5` | 7 | 203,282 |

Status at 22:38 (47 min in): generation done in all shards, **14,670 records generated**
(s1 2,934 · s2 2,970 · s3 2,970 · s4 2,934 · s5 2,862); reviewer (Gemma) loaded in 5-9 min; review at
~13 records/min per shard (112-167 done each), so ~3.5-4 h more, expected end ~02:00-03:00. The shards
review with `repetition_penalty` 1.05, so expect ~25% malformed reviews as in pilot 2; redo them after the
fix (review only, no regeneration). The test job 59928215 was still pending (our GPU share is in use).

Check: `sacct -j 59925723,59925724,59925725,59925726,59925727 -X -o JobID,State,Elapsed` and the stage
lines in `slurm/logs/dk-gen-gated-<job>.out`. A failed or timed-out shard: rerun the same submit command
(it reuses the shard seed dirs, generate/review resume):
```
source $WORK/alembic/me.env; cd $REPO
export SBATCH_ACCOUNT=EUHPC_D30_031 CONFIG=configs/gen_hpc10k.yaml; source slurm/env.sh
SHARDS=5 SEEDS_ALL=runs/hpc10k/seeds RUN_PREFIX=runs/hpc10k_s bash slurm/submit_gated_shards.sh
```
(It resubmits all 5; cancel the ones that already finished, or `RUN_DIR=runs/hpc10k_sN sbatch
--time=24:00:00 slurm/generate_gated.sbatch` for one.)

**Next steps**
1. Fix the early-stopping reviews (see pilot 2), test on `runs/hpc10k_pilot100`, then re-review the shards.
2. When the shards finish: merge `runs/hpc10k_s{1..5}/clean.jsonl` → `runs/hpc10k/clean.jsonl` with a
   cross-shard near-duplicate pass. The gold-example source docs (`slurm_sbatch_basics`,
   `slurm_monitoring`) are copied into every shard, so their records repeat with the same ids: keep one copy.
3. Write per-gate counts (per shard and total) and a 100-record spot-check sample into PR #31 and
   `docs/TRAINING_PROGRESS.md`.
4. Batch 5: train on 4,775 + new clean data (batch 2 settings, rank 16, 3 seeds); compare with 3b.
5. Still to add to `TRAINING_PROGRESS.md`: per-step dev evals of batch 3/r8 (jobs 59920008, 59920016,
   59920263) and batch 2 seed 2 step 284.
