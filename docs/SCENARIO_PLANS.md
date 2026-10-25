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

## Optional source input in v2 plans

`source-scenario-plan-v2` adds an optional `context_spans` list to each entry.
All v1 fields are still required. V1 continues to reject this field and retains
its exact entry-digest formula and IDs; the legacy grid is unchanged. V2 entries
use `scenario-v2/` IDs bound to their version, full entry and resolved persona.
Changing source input changes the ID; it does not prove a new learning objective.
An entry with no context keeps the existing question and tool behavior.

Each explicit span has only `start`, `end` and `sha256`. For example, the following
is an illustrative entry-field fragment, not a runnable plan:

```json
{
  "context_spans": [
    {"start": 40, "end": 83, "sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
    {"start": 120, "end": 250, "sha256": "0000000000000000000000000000000000000000000000000000000000000000"}
  ]
}
```

Offsets are strict integer **character** positions, with an inclusive start and
exclusive end, in the exact normalized text from `seeds.load_chunks(cfg)`.
They are not byte offsets or positions in the original Markdown file. SHA-256
hashes the exact UTF-8 substring. The list must be nonempty when present, ordered
and nonoverlapping. Empty/whitespace spans, invalid bounds, changed hashes and
forbidden or unresolved training sources fail before a teacher client exists.
Context is currently permitted only for `prose`, so it cannot supply tool targets
or alter action authority or intentionally missing arguments in clarification
examples. V2 tool entries without context remain supported.

Select the smallest sufficient problem input: code being examined, table values
and headers, and explicit preconditions. The generator never selects all code
fences, removes comments or metadata, fills boilerplate, or invents facts. Plain
preconditions and bounded code fragments do not need to be whole programs. When
a selected span contains line-start Markdown fences, its fences must balance;
quoted/commented markers inside a plain code fragment remain data. This is a
narrow structural screen, not a Markdown parser or language/compiler check.

The `source-input-context-v1` renderer puts each exact span in a **separate**
numbered quoted-data block (`Provided input 1`, `Provided input 2`, ...). It never
concatenates separate programs as one program. Each outer backtick fence is
longer than any backtick run in its span. Source bytes are unchanged inside each
block; ordinal labels and block separators are visible formatting only. The
question row's `scenario_context` binds the version, span offsets/hashes/text and
the SHA-256 of the complete rendered input. These fields also accompany answers.

Only context-enabled questions use
`source-grounded-v10-bound-input-context`: the question teacher sees the exact
selected input and is told it will be attached automatically. The stored
`question` is the stripped teacher question followed by the deterministic input
blocks. That identical string enters the actual answer prompt and the student's
stored user message. The full source still grounds the answer teacher; the
scenario brief does not enter its prompt. Source text is quoted data, never a
system instruction. No teacher answer or evaluation reference is inserted.

`question_generation` preserves the original unstripped parsed question, the raw
text returned by `Teacher.chat_json`, both hashes, capture version and composed
question hash. This is not a provider HTTP-envelope/finish-reason claim; campaign
capture remains separate. Metadata is not used as a student message or target by
`records.training_example`. Existing answer and prose-logprob formats stay intact.

For context-enabled questions, `question-code-fence-v2-raw-and-composed` checks
the raw teacher question and effective question independently. An appended input
block must not conceal a truncated raw question. Rejections retain the exact
candidate and captured raw text; they are not regenerated, repaired or converted
into artificial answer rows on resume. Campaign capture validators must use the
effective per-record check version and both-text check, rather than recomputing
only `_question_issue(candidate['question'])`. Planned-answer counts still include
answers blocked by a rejected question.

Resume checks reconstruct the input and composed question, verify raw-response
parsing and capture hashes, and compare the actual cached answer's user turn
before a client exists. A context-enabled cached prose answer must retain the
production two-turn user/assistant shape, with no extra user/system turns or tool
fields. The run manifest binds context rendering/capture/check
versions and prompts as well as plan/source/code. A changed span, prompt or code
requires a fresh run directory; copying a manifest is not a valid migration.
These consistency checks do not authenticate adversarially rewritten output plus
all of its hashes; preserve the independent campaign capture and artifact hashes.

Source hashes cannot establish that a selected span is appropriate input. The
agreed model-review process must exclude answer keys, `@@expect` metadata,
output-revealing comments, reference answers and explanatory conclusions that
give away the requested answer. There are no arbitrary text, answer or hidden
label fields in the span schema. Review generated questions against their actual
attached inputs for premise validity, relevance, completeness and novelty.

Token preflight must include all rendered input blocks in the outgoing question
request, effective user question and answer request; reserve the teacher's
question output and student's response budget. Do not relax truncation/loader
limits to fit overly broad excerpts. The actual Q builder is
`generate._question_prompt(job, scenario)`; `scenario_plan.context_block(job)`
provides the exact appended input for budget calculations. These helpers make
preflight match production; they do not run a model. No real-model improvement or
semantic completeness is established by the deterministic feature or its tests.

`tests/test_source_context_inputs.py` exercises actual generation, source
admission, planner, schemas, appenders and the record adapter with synthetic
fixtures and a fake teacher. It covers legacy IDs/prompts, separate spans,
Unicode/newline positions, input/hash/holdout failures before requests, raw
capture, actual prompt/user positions, tool-context exclusion, rejected-question
preservation and exact/partial/stale resumes. Run the focused and full suites in
the existing compatible dependency environment; no generated code is executed:

```sh
PYTHONPATH=src python -B -m pytest -q tests/test_source_context_inputs.py tests/test_scenario_plan.py tests/test_prose_answer_scope.py tests/test_source_registry.py
PYTHONPATH=src python -B -m pytest -q
```

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
