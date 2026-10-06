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
