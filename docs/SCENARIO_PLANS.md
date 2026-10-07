# Explicit source-bound generation plans

`generate.scenario_plan` opts into a JSON plan of reviewed scenarios. Leave it
unset to retain the existing persona/task grid, sampling, IDs and resume behavior.
This feature does not generate training data by itself or establish semantic
novelty, source support or licensing permission.

Each entry produces one question through the existing question teacher, followed
by `generate.answers_per_question` answer attempts through the existing prose or
tool-trace path. The plan determines modes directly; `questions_per_chunk`,
`tool_fraction` and `tool_mix` do not expand or resample it. For example, 20 entries
and two answers per question plan 40 candidate answers, not 40 accepted examples.
`generation_manifest.json` records these counts and the actual mode distribution.
Plan the approximate 70% prose / 30% tool share using meaningful scenarios.

## Plan schema and source admission

Set `seeds.registry` to the reviewed source-family registry and
`generate.scenario_plan` to the JSON file. Relative paths use the process working
directory, as other paths in the existing config do. Every field below is
required; additional fields, including expected answers, are rejected.

```json
{
  "version": "source-scenario-plan-v1",
  "entries": [
    {
      "scenario_id": "count-versus-cause",
      "round_id": "reviewed-batch-01",
      "doc_id": "synthetic_example",
      "chunk_id": "synthetic_example#0",
      "document_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "chunk_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "persona_index": 0,
      "task": "concept",
      "mode": "prose",
      "capability": "distinguish-observation-from-cause",
      "brief": "Ask whether the source's stated count alone establishes a cause, without inventing an error or diagnosis."
    }
  ]
}
```

This is a non-runnable synthetic format example. Replace the document, chunk,
hashes and brief with reviewed training-source content. `document_sha256` hashes
the original source file bytes. `chunk_sha256` hashes the exact UTF-8 `text`
returned by `seeds.load_chunks(cfg)`, after existing paragraph packing. It is not
a hash of an arbitrary excerpt or a re-rendered document.

Only an admitted training chunk can resolve. Explicit `eval_docs`, held-out
family ancestry, unresolved admission, changed file bytes and changed chunk
boundaries fail before the teacher client exists. A non-prose entry also needs
its document in the explicit `generate.tool_doc_ids` list. The persona index and
task must resolve in the configured lists. Allowed tasks are `concept`, `howto`,
`debug`, `script`; modes remain `prose`, `call`, `ask`, `none`.

Use a stable, descriptive `scenario_id` and a documented `round_id`, both lowercase
slugs of at most 80 characters. A round is an explicit reviewed batch, never a
multiplier. Duplicate round/ID pairs fail. Normalized identical briefs for the
same source chunk, task and mode also fail even if the entry changes its ID,
round, persona or capability label. Changing task/mode or paraphrasing a brief
does not prove novelty; review scenario coverage and apply global question
deduplication across retained rounds before accepting data.

Question IDs contain the version, round, scenario ID and full SHA-256 binding of
the entry and resolved persona. Entry order does not change IDs. Content changes
do change IDs. The brief and capability enter the actual question prompt as
planning input, explicitly subordinate to source and tool evidence. They never
supply a reference answer, and do not enter the answer prompt. Other plan fields
remain provenance metadata. Existing messages, tools and prose logprob formats
are unchanged. Tool mode behavior remains the existing production behavior;
this extension does not add tools or make unsupported source/tool pairings valid.

## Fresh runs and exact resumption

Before any teacher client or output appender, scenario generation writes a new
exclusive `generation_manifest.json` with UTC creation time and a content binding
over the effective seed/generation/teacher settings, source and registry hashes,
plan bytes, gold bytes and rendered gold, prompts/tool schemas, generation code,
valid flag set, resolved jobs and planned counts. Source bytes from evaluation
documents are never copied into prompts or this manifest; their file hashes and
registry assignments are exclusion and change-detection inputs only. No
evaluation questions or reference-answer files are read by this feature.

The manifest omits teacher endpoint routing and the authentication environment
variable name, and never reads credentials or environment values. An allocation
can resume at a different server port. It also omits unrelated train, evaluation,
judge and export settings. The configured teacher model and generation settings
remain bound. Record the actual pinned model snapshot and serving environment in
the campaign run manifest: an unchanged model alias does not establish unchanged
weights. Do not put credentials into ordinary generation settings.

Any changed binding requires a new, unused run directory. This includes editing
or reordering the plan, changing gold examples, source content, chunking, persona,
sampling, model or prompt code. Do not copy or rewrite the old manifest to make a
changed run resume. Pre-existing generation/verification artifacts without a
matching manifest are rejected and preserved. Exact resumption validates cached
source/question/answer bindings, skips completed answer IDs and generates only
missing work. Stale or malformed cached inputs fail closed before teacher access;
they are preserved for diagnosis rather than silently repaired. The manifest is
an input binding, not a cryptographic proof that someone has not altered outputs.
Use one generation process per run directory.

## Validation and limits

`tests/test_scenario_plan.py` exercises the real planner and generation entry
point with synthetic fixtures and fake teachers, including both prose and tool
paths, manifest-before-client/writer ordering, counts, IDs, source exclusion,
changed inputs, partial resumption and stale cached markers. Fake completion and
logprob data are test fixtures, not real teacher outputs or quality evidence.

Run with the existing compatible Python 3.12 project environment:

```sh
PYTHONPATH=src python -B -m pytest -q tests/test_scenario_plan.py tests/test_source_registry.py
PYTHONPATH=src python -B -m pytest -q
```

These automated checks cannot determine whether a brief is meaningful, copied
from a held-out answer, semantically duplicated or unsupported despite its
declared training source. Reviewed planning inputs and downstream source-support
review remain necessary. Admission fields record reviewed provenance decisions;
neither hashes nor this planner grant permission to use a source. Candidate
counts do not predict acceptance, training benefit or practical throughput.
