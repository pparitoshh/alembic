# Evaluation data

`eval_all_v2.jsonl` is the normal **development evaluation** release: the ten
`eval_core.jsonl` rows, sixteen `eval_req.jsonl` rows (including `req-016`), and forty
supplied model-reviewed supplemental rows, each exactly once. Original slices
and the original 26-row `eval_all.jsonl` are preserved. Tool evaluation and judge
calibration remain separate datasets with separate schemas.

The release has been examined during development. It is not a newly blind final
test. Keep its questions, references, paraphrases and both source documents out
of training, gold examples, preference data and retrieval. Use development failure
categories to guide independent training coverage; do not turn these answers into
training targets. Fresh final testing requires separately held-out source families.

The agreed acceptance method is model review plus automated validation. No human
review is claimed or required. Exact original generator/reviewer model versions
are unknown. The manifest binds input and output bytes, source hashes and supplied
review reports. The later integration review read all originals, resolving the
earlier supplement review's missing-`req-016` limitation. Source licensing/origin
gaps remain unresolved; this release does not resolve them.

Schema, unique-ID and exact/normalized-question checks pass. Normalization uses
Unicode NFKC, casefold and collapsed whitespace. These checks do not detect all
semantic duplicates. Model review retained related applications, including:

| Related items | Limitation |
| --- | --- |
| arr-cand-013 / arr-003 | Both concern array log identity; the candidate diagnoses collisions across submissions. |
| arr-cand-005 / arr-002, arr-003, arr-010 | Combines submission/task identity and explicit file-position mapping. |
| Array dependency items | Several apply the same small set of dependency primitives. |
| req-cand-002, 014, 015, 018 / req-016 | Different applications share requeue identity preservation. |
| req-cand-006 / req-cand-017 | Both use hold/update/release; one adds selective release of two jobs. |

The 66 items are not 66 independent capabilities. Report capability-level results
and limitations alongside item totals. Script fragments and explicit assumptions
must be judged as written; `bash -n` is neither Slurm validation nor execution.

Compare base and trained models on the same frozen dataset and protocol. Do not
compare aggregate scores from the original 26 and expanded 66 as if they measured
the same sample. References go to the judge, not the closed-book answering model.

The tool evaluation remains a draft with model review pending. Calibration tests
the judge; it is not evidence of student accuracy.

## Eval v3: trained-topic development set (200 questions released)

The frozen 66-row `eval_all_v2.jsonl` and its scoring protocol are unchanged.
`eval_v3.holdout.json` freezes source exclusions before candidate generation, with
text hashes, original chunk IDs, upstream URLs, licenses and Slurm version.
Eight bundles in `v3_sources/` target 200 normal QAs across dependencies (30),
accounting (30), job control (25), output/mail (25), resources (30), queue/cancel
(20), multi-constraint scripts (20), and debugging (20).

**Released:** `eval_v3.jsonl` contains 200 questions, with hashes and counts in
`eval_v3.manifest.json`. From 512 original candidates, 368 passed source review
and verification, 353 passed question-quality review, and 200 filled the fixed
quotas. Tasks: 63 how-to, 46 debug, 57 script and 34 concept.
The build requires independent source support review, deterministic verification,
a separate question-completeness review, exact/normalized/lexical near deduplication,
and all topic quotas. Selection never uses student scores. Review is automated;
no human review is claimed. Failed or uncertain reviews cannot release a row.
References cite their bundle chunk; the evidence sidecar resolves original chunks.

`decontam` now also checks `*.holdout.json`. Training admission rejects matching
sources even if an old verified file is reused. These protections apply only in
checkouts that contain this change and the manifest. The teammate-owned live
Leonardo checkout is read-only to this account: **apply this PR before merging or
training B7**, or use the isolated patched checkout described in the runbook.

The completed post-release audit found:

| Historical input | Rows | Source matches | Question matches | Excluded union | Remaining |
|---|---:|---:|---:|---:|---:|
| hpc10k | 10,773 | 910 | 0 | 910 | 9,863 |
| Eight top-ups, summed (not deduplicated across runs) | 10,248 | 1,300 | 0 | 1,300 | 8,948 |
| B6 training, historical exposure only | 12,773 | 910 | 0 | 910 | 11,863 |

Question matches use exact/normalized text and word-trigram Jaccard at 0.8 against
the evaluation files, including the final v3 questions. Zero lexical matches does
not establish absence of semantic paraphrases. Job `60045496` wrote new filtered
copies under the isolated checkout's `runs/eval_v3_audit/`; original inputs were
not changed. None of these numbers represents an already-clean merged training set.
Historical B6 exposure cannot be undone: this is a development set, not blind
testing or evidence of generalization to unseen source text. A genuinely blind
test still needs separately reserved source families.

Evidence: `research/eval_v3/decontamination_audit.json` (per-input final counts and
hashes), `research/eval_v3/release_evidence.json` (all 200 selected source and
quality reviews), and `research/eval_v3/source_exposure_before_release.json`.
Build instructions
and job paths: `research/eval_v3/README.md`. Report v2 and v3 separately, including
per-topic results; shared sources mean an independent-item ±0.07 margin is only
a rough approximation, not a validated uncertainty bound.

Base/B6 v3 comparison is pending: the three B6 weight files are owner-only and
unreadable to the evaluating account. See the runbook for the exact access fix.
