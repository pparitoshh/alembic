# HPC evaluation dataset v1

Internal, model-reviewed release: **151 tasks** owned jointly by Vaibhav Mangroliya and Shraddha Jadhav. This extends the integrated development RC without changing its 82 records. No inference or student improvement is claimed by dataset construction.

| Slice | Ready | Scope |
|---|---:|---|
| `development_normal.jsonl` | 106 | Existing 66 Slurm arrays/requeue items plus 40 Triton GPU, MPI, environment, scheduling and performance items |
| `development_tools.jsonl` | 23 | Existing 16 plus 7 first-response decisions and arguments |
| `final_normal.jsonl` | 22 | 12 CUDA Samples and 10 Open MPI source-family-held-out tasks |
| Judge calibration, separate existing file | 24 pairs | Never included in the 151-task total |

There are zero unresolved content drafts in this selected release. Content readiness does not mean comprehensive HPC coverage, runtime correctness or complete licensing. `coverage.json` gives exact source/task/topic counts and limitations.

## Evidence and review

`item_metadata.json` binds each question/reference or tool expectation to source hashes, evidence spans, scope and review lineage. New primary-source bytes and applicable notices are retained under `sources/` and `notices/`; upstream URLs, revisions and retrieval dates are in `source_manifest.json`. Aalto examples describe the pinned **Triton** environment, not Leonardo. CUDA Samples support Toolkit12.9 at the pinned revision; acquired Open MPI is **5.0.8rc3/gitclone**, not a final5.0.8 release. Version-specific sample behavior must be scored within that scope.

New items received author model review, a separate model-agent evidence review and coordinator acceptance, followed by automated checks. These are fallible model reviews with shared construction exposure, not independent model-family votes or human review. Exact underlying reviewer-model version was not available and is recorded unknown. Original draft sidecars preserve their pending-at-creation state; `review.json` records the subsequent acceptance. Item `final-cuda-005` was narrowed before freeze to remove incidental internal-constant recall; `changes.json` preserves both versions.

The original six Slurm source notes retain **18 unknown URL/version/license fields**. New source provenance does not repair those gaps. This is an authorized internal repository handoff, not an externally published benchmark or dataset-license clearance.

## Splits and scoring

Existing development exposure remains disclosed. Aalto-Triton is reserved for development; the Open MPI and CUDA Samples upstream families, including related versions/copies, are reserved for final evaluation before new training generation. The 22 final items have model construction/review exposure and unknown pretraining exposure; they are not claimed to be unseen by everyone. Once frozen, do not use these items, answers, source chunks or failure results for iterative training-data or prompt changes. Use development failures for iteration.

Normal files retain the five-field project schema. References permit supported paraphrases and equivalent snippets; they are construction/scoring evidence, never inserted into closed-book answering prompts. Code fragments were source-reviewed but **not compiled or executed**. No executable-correctness score is established here.

Tool items use `tool-first-response-v2` with visible `hpc-tools-clarify-first-v1`. Score decision, tool name and schema-valid arguments for at most one first-response call. Explicit queue/resource discovery is permitted where stated; missing/ambiguous job selection requires clarification. Submission/cancellation cases use isolated mocks only. No-call passing does not prove useful clarification or truthful prose. First-response passing does not measure multi-turn completion, final-answer grounding or real Slurm execution. Common visible tool schemas are the shared application interface; protected scenario/label/reference content must not become training data.

## Reproduction and compatibility

Run `python data/eval/dataset_v1/validate_release.py` from the repository. Run `pytest tests/test_eval_dataset_v1.py` with existing project dependencies for production-scorer regression fixtures. These commands make no model requests and execute no generated workload; mock submission uses `bash -n` only.

`config_overrides.json` uses actual existing configuration keys and deliberately keeps development and final selection separate. Original defaults are unchanged. Do not concatenate the old26-row `eval_all.jsonl` again. Calibration and diagnostic fixtures remain outside these datasets.

All82 old baseline answers remain preserved. Normal66 answering inputs remain compatible for later consistent versioned rescoring. Old16tool responses are incompatible with the changed visible policy inherited from dev_v3. New69items have no cached answering outputs. Nothing was rescored or regenerated in this release.

Remaining gaps include fresh final Slurm/tool/performance tasks, broader CUDA/MPI/profiling coverage, actual compiled checks and multi-turn tool evaluation. See `manifest.json`, `validation.json`, `review.json`, `coverage.json` and `cached_answer_compatibility.json` for exact bindings and scope. No training records are added.
