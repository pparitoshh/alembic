# Reviewed multi-origin training inputs

`train.input_manifest` is an optional `{path, sha256}` object. Its hash binds a
reviewed `multi-origin-training-inputs-v1` JSON manifest. When unset, training
continues to read `run_dir/verified.jsonl` and uses the original single-registry
admission behavior. Generation and verification do not use this new option.

The multi-origin path runs inside `train.admitted_training_rows`, called by
`train.run` before importing the training runtime or loading a tokenizer/model.
It returns the original raw records in manifest order. It does not replace a
record's historical registry hash with a combined or newer registry hash.
Adapter-ready or tokenized rows are not an alternate input format.

## Manifest and immutable inputs

The strict types in `src/distillkit/training_inputs.py` define the complete
schema. Unknown keys are errors. The top-level fields are:

| Field | Meaning |
| --- | --- |
| `version` | Exactly `multi-origin-training-inputs-v1` |
| `root` | Input root, absolute or relative to the manifest file |
| `input_bindings` | Every consumed file's root-relative path mapped to its SHA-256 |
| `expected_rows` | Exact number of selected records |
| `contract` | Common student, system-prompt hash, maximum length, current `records.py` hash and optional raw tool-schema hash |
| `origins` | Ordered original-registry or explicitly certified legacy subsets |

Every origin binds its unique `origin_id`, exact `record_ids` and `count`, raw
`records`, original `generated` and `questions`, original `config`, `run_root`,
`run_manifest`, `source_manifest`, generation revision, selected seed filenames,
gold files with hashes, and original CPU evidence. Subset lines must be
byte-identical to the selected original generated lines. JSON duplicate keys,
duplicate IDs, blank JSONL lines, nonfinite numbers and incomplete final lines
are refused. IDs and normalized questions must be unique across the union.
Question normalization is Unicode NFKC, case folding and collapsed whitespace;
it is not a semantic deduplication detector.

An `original-source-registry-v1` origin additionally binds its original
`registry`, `generation_manifest`, `release_manifest`, `certificate` and raw
acceptance `sidecars`. Current supported certificate forms are the existing
`DELIVERED_UNDER_DOCUMENTED_CHECKS` and
`training_eligible_under_recorded_checks` forms. Each must bind the exact
subset, selection manifest and passing CPU artifact. Per-record sidecars must
bind the original record, source, configuration, registry and release, with
recorded quality/training acceptance. This reuses documented evidence; it does
not rerun or manufacture a semantic review.

Relocation changes only in-memory `seeds.dir`, `seeds.registry` and
`generate.gold_dir`, resolving `run_root/corpus/{seeds,registry.json,gold}`.
The immutable original configuration is parsed directly, without calling
`load_config` or loading dotenv. Its effective generation settings and registry
are checked against the original generation manifest. The latter's content hash
also preserves its recorded template/code/source bindings. Original teacher
identity and settings are checked; transport address and API-key *variable name*
are excluded from the generation-settings comparison as in the existing
generation manifest. Credentials are not input artifacts.

All `corpus/` bindings in the original run manifest must be available unchanged,
including permission notices, attribution and review evidence. Declared files
must resolve inside the input root; relative escapes, symlink escapes and
`.env`-prefixed path components are refused. Input hashes are checked again
after the production helpers read the files. The manifest root itself is an
explicit operator-selected location, not a sandbox or a source of permission.

The real `admitted_sources`, `_gold`, `load_chunks`, `source_for_record` and
`records.training_example` helpers run against each origin. The union also
reconciles all recorded family declarations, transitive ancestry, explicit
evaluation IDs and known held-out content hashes across origins. A holdout in
one origin cannot silently become training material through another origin.
Conflicting declarations fail rather than being resolved by input order.

## Narrow legacy compatibility

A `legacy-question-chunk-certificate-v1` origin has `registry: null` and binds
its original `legacy_review`, `legacy_review_bindings` and CPU evidence. Every
input named in that review must map to an exact hash-bound file, including the
original questions and generated records. Its explicitly eligible record IDs
must equal the selected subset, and the recorded quality/pipeline decisions,
whole-source hashes, reconstructed chunk hashes and permission/split evidence
must agree. The original effective config may differ from the run snapshot only
in the teacher's ephemeral `base_url`.

In addition, every legacy training source, including any used only by gold,
must match the same document ID and exact source bytes in a registered origin
of this manifest. This corroboration is reported separately from historical
provenance: the loader refuses retroactively added registry fields on legacy
rows. An uncertified unregistered origin, or a legacy-only manifest with no
registered corroboration, is refused. This path contains no campaign-specific
record IDs or hashes. It cannot establish unknown upstream licenses, versions
or authorship, and it does not repair unresolved historical provenance.

## Training contract and evidence

The caller and all originals must agree on student model, system prompt and
maximum length. The current raw-record adapter file must match the reviewed
contract. Nonempty raw tool schemas must match the declared common schema hash;
prose rows may have no schemas. Raw tool argument dictionaries remain unchanged;
the existing adapter converts them to JSON strings only in its copied training
example. Diagnostic/ineligible records and runtime-guard targets remain refused
by that adapter. A changed common contract requires renewed combined checks and
a reviewed manifest; it is not an automatic migration.

On `train.run`, successful admission writes `training_inputs.lineage.jsonl` and
`training_inputs.admission.json` in the training run directory. Each lineage row
records the original input path/line hash, record hash, original config and
registry, generation revision, gold bindings, inherited evidence and actual
adapter-output hash. Existing evidence must be identical; it is never appended
to or overwritten. Direct CPU callers may request the same evidence using
`admitted_training_rows(cfg, evidence_dir=...)`, or omit it for a read-only check.
Use a fresh training run directory for a changed input manifest.

The manifest and historical certificates are reviewed trust inputs, like the
source registry. Hash binding is not an authenticity signature, factuality
proof, new permission decision or semantic-leakage detector. A caller capable
of replacing all reviewed inputs can also alter a registry's admission flags.
The loader does not rerun support review, tool replay or token/mask/length checks,
and does not authenticate arbitrary historical CPU assertions. A production
release still needs those independent checks and a recorded reviewer/approval
lineage. Model-name equality is not a tokenizer snapshot pin; record the actual
tokenizer/model revision and runtime before a real training run, and recheck
token lengths, assistant-only masks, EOS and tool serialization there.

## Tests and compatible environment

Run without bytecode creation, using the existing project environment and its
installed development dependencies (Python 3.12 according to `pyproject.toml`):

```sh
PYTHONPATH=src python -B -m pytest -q tests/test_training_inputs.py tests/test_source_registry.py tests/test_training_arguments.py
PYTHONPATH=src python -B -m pytest -q
```

The new tests use synthetic evidence and real production admission/adapter
functions. They forbid teacher construction, dotenv reads and network access.
The `train.run` tests stop at the first training-runtime import after admission,
or assert that invalid inputs fail before that point; no model is constructed.
They cover original and current certificate shapes, relocation, legacy scope,
source/gold/config staleness, cross-origin holdouts, duplicates, schema/adapter
compatibility and preservation of original bytes. A passing synthetic test is
not a real tokenization, teacher-inference or student-training result. No
dependency installation is required by this change.
