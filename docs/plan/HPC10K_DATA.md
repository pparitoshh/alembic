# Plan: ~10k quality-gated Slurm / cluster-usage training records

Status: draft, 2026-10-10. Branch `feat/data-hpc10k`. Results so far live in
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
