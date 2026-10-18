# AI Factory pilot — local experiment, not a model release

Owner: Vaibhav Mangroliya (Hugging Face destination: `VaibhavKKM`). Built on Team Alembic's DistillKit framework and Qwen's base/teacher models. No model has been uploaded by this experiment.

## Scope

Train only on admitted official technical documentation for cluster operation: Slurm jobs and troubleshooting, GPU/MPI workloads, software environments, containers, storage/quotas and access. The downloaded EU archive is a collection, not an approved training dataset. News, marketing, unrelated libraries, untranscribed videos and material without adequate provenance are not automatically training inputs.

The first pilot uses 48 admitted LUMI pages and reserves 3 complete pages before generation. It plans 256 candidates; this is not a verified-row count. Retrieved facts remain site-labelled: LUMI settings must not be presented as Leonardo settings. Retrieval additionally uses original Slurm and CINECA technical pages.

## Execution

1. GPU smoke job tests both Python environments and an actual vLLM model response.
2. Independent jobs run the base-vs-base+documents benchmark and teacher generation/source review concurrently.
3. Generation must pass independent Gemma source support, deterministic checks, deduplication, frozen evaluation exclusions and exact training-token-length checks. Training requires at least 100 accepted balanced rows with all tool modes represented; thresholds are not relaxed for throughput.
4. Three BF16 LoRA runs (seeds 42, 43, 44) depend on successful data preparation. They all consume an identical hash-bound release. Source pages were reserved before generation, so the pilot has no additional seed-dependent train/validation split.
5. Each seed runs frozen v2/v3 and tool comparisons plus paired-order judging. Each judge must pass the existing synthetic calibration smoke gate. Aggregate results report seed spread separately from question-bootstrap uncertainty.

Every GPU task requests one GPU and one schedulable node; Slurm can place independent tasks on the same node or different nodes. Up to 3 training seeds plus the independent retrieval task can run concurrently when allocated. No reservation is assumed. No shared checkout or model environment is changed.

## Interpretation and next gate

This is an initial experiment, not the full EU AI Factory model. The lexical retrieval baseline is not optimized and needs evidence inspection. It indexes public documentation, including held-out-topic documentation, never evaluation questions/answers. Its scores are document-assisted and must be labelled separately from closed-book scores.

Frozen HPC v2/v3 can reveal regressions but cannot establish quality on all AI Factory services. Before scaling or publishing a quality claim: independently review a new AI Factory test set from reserved sources; inspect command correctness, citations, site confusion and abstention; compare student+retrieval as well; measure quantization effects. Avoid selecting a winner from one seed or calling win rate accuracy. Preserve source and team credits. No 99%/100% guarantee is made.
