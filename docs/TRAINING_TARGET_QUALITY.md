# Training target checks

Source-supported text can still teach an incorrect operation. In the local
5,147-row development experiment, a targeted audit found nine answers that used
`sattach --error-filter` where the question required stdout filtering, plus one
answer with `--distribution=plane,4`. All ten had historical source-review
approval. This targeted finding is not an estimate of the dataset's total error
rate.

Training admission now checks fenced shell commands before loading model weights.
It flags these two failure patterns and writes `training_target_quality.json`
when an evidence directory is supplied. A flagged record blocks training until
it is quarantined or corrected and independently reviewed in fresh inputs.
The guard does not rewrite answers or modify frozen experiments.

The check is deliberately bounded: it does not validate arbitrary shell
programs, option values generally, inline examples, task completeness or all
Slurm semantics. A valid stderr filter remains valid for a stderr question.
Pedagogical answers containing intentionally broken executable examples may need
review. Passing this check is not a new semantic verification stamp.

Validation splitting now keeps records from the same `source_family` or
`source_families` together, including transitive connections through multi-source
records. Without family metadata, document-based behavior stays unchanged. This
prevents two excerpts from the same declared source family crossing the internal
training/validation boundary. It does not replace the frozen evaluation-source
exclusion audit. Reserve validation source families before generating the next
dataset and use the same reserved set across compared training seeds.

These are data-quality and measurement fixes, not a claim of higher model
scores. The next model comparison must still use unchanged questions, matching
prompts and decoding, paired answer-order judging, and all declared seeds.

Command semantics were checked against the [Slurm sattach manual](https://slurm.schedmd.com/sattach.html),
the [Slurm 25.05 option parser](https://github.com/SchedMD/slurm/blob/slurm-25-05-0-1/src/sattach/opt.c),
and the [srun distribution syntax](https://slurm.schedmd.com/srun.html).
