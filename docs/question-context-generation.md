# Self-contained question context

The question author sees a source chunk and may see a scenario brief. The
student's stored user turn contains only the generated question. These are
different contexts: source support for a numerical answer does not make a
question about an absent table answerable, and an exact source rewrite is not
a useful training target when the user never supplied the function or bounds.

Prompt version `source-grounded-v8-self-contained-resource-policy` introduced the rule that asks
the author to include necessary table values, code, bounds and assumptions in
the question, or choose a narrower objective. It forbids treating unseen
examples as user-supplied inputs and copying the reference answer into the
question. Teacher, answer settings, sources, gold examples and data schemas
are unchanged. The answer prompt's version label changes with the shared
version constant; its substantive answer instructions do not change.

Production-path fake-teacher tests check that the rule reaches prose and
call/ask/none question requests and preserves source/gold/holdout boundaries.
They do not prove the real teacher obeys it. Every generated question still
needs an evidence-grounded premise/context review before acceptance. No
historical question or answer is repaired or relabeled by this change.

Use a fresh generation directory and record the new prompt/config/code hashes.
The existing immutable pilot remains attributable to its original version.

## Scoped prose answers and complete fenced context

Version `source-grounded-v9-scoped-prose-complete-question` adds instructions to
the actual **prose** answer prompt: answer the requested scope, prefer the
smallest required fragment, and do not add an unsolicited full program or setup,
API constants, allocation semantics, helper calls or computation absent from the
source and request. Preserve source qualifications and existing computation
placeholders. Mathematical equalities should not be labeled as shell assignments.
The tool-policy rules, tool schemas, teacher settings, sources, gold examples,
training-row schema and logprob representation are unchanged. The shared prompt
version label also changes in tool requests, but the prose-only answer instructions
are not injected into the tool conversation path.

The question prompt reinforces short, complete, self-contained code context,
including real line breaks and closed fences. Narrow the objective instead of
cutting off the necessary snippet. This guidance is not evidence that a teacher
will comply: pilot review exposed valid JSON containing truncated fenced code,
unsupported full-program scaffolding and missing source-only context.

The `question-code-fence-v1` screen detects an unmatched run of at least three
backticks or tildes before answering a stored question. It also recognizes
fences joined to introductory prose, since a JSON-authored question can omit
the expected newlines. A closing marker must have only whitespace elsewhere
on its line, even when the opening was joined to introductory prose. There is
no end-of-question exception: a terminal marker inside an unfinished string or
comment must not make an unclosed block pass. Embedded runs in
code strings such as `printf("```")` or comment text are not closing lines.
Single/double inline backticks are not fences, escaped markers are literal,
and a shorter/different marker inside a block does not close it.
This is a deliberately narrow structural screen, not a Markdown
parser, syntax checker, finish-reason check, or detector of every truncation,
unsupported premise or missing-context reference. Balanced fences do not prove
that their contents are complete or correct. Literal runs of three or more
markers outside those conventions, including a closing marker joined to code
on the same line, can be conservatively flagged. The prompt asks for proper
line breaks; this screen does not repair them or parse each programming language.

A rejected question remains in `questions.jsonl` under its original ID. For a
newly generated rejection, `question_validation` preserves the exact original
question string and the raw text returned by `Teacher.chat_json`, their SHA-256
hashes, check version, status and machine-readable reason
`unmatched_question_code_fence`. This is the teacher abstraction's returned
text, not a claim to preserve the provider's raw HTTP envelope or finish reason;
the campaign capture remains separate. Existing cached rows without these raw
fields are preserved as received; unavailable raw responses are not invented.

`question_rejections.jsonl` records the full captured candidate, its canonical
JSON hash, ID, check version and computed reason. The screen is recomputed from
the question text; annotations or an existing audit row cannot authorize an
incomplete question. Capturing the reason/raw text in the question row first
allows an interrupted audit append to recover without a new teacher request.
Identical resumes neither duplicate the rejection nor regenerate the question.
No rejected question is answered, rewritten, deleted or counted as invalid JSON.
No artificial answer/training row is created for it.

For counts, distinguish planned questions, captured parsed questions,
rejected-question candidates, questions eligible for answers and completed
answers. For example, two planned questions with one structural rejection and
two answers per question produce two captured questions, one rejection and two
completed answers against four planned answers. Campaign wrappers must retain
that partial outcome and the intentionally missing/pending answer IDs; a
successful generation function return is not proof that the planned dataset is
complete. An all-rejected run can have no `generated.jsonl`, as with an all-invalid
JSON run. Preserve its questions and rejection audit rather than manufacturing
an empty-answer record or silently retrying it.

Scenario-plan manifests bind the question-check version, full prompt content and
generation code before requests. The rejection artifact is also an existing-run
artifact: it cannot be adopted into a fresh manifest. Use a fresh run directory
for a changed prompt/check/source/config. Legacy grid runs retain their prior
ID/resume behavior, with the same fence screen applied to cached question text;
existing outputs are not repaired or relabeled.

`tests/test_prose_answer_scope.py` uses actual `generate.run`, `Config`, source
registry/planner, schemas and JSONL appenders with a fake teacher. It checks the
outgoing prompt positions, preserved answer/logprob format, valid fences and
inline backticks, missing-ID clarification, rejected raw-text hashes, mixed-plan
counts, crash recovery, cached-row screening, identical resume and fresh-run
manifest failures. These tests establish integration and deterministic screening,
not real-model factual support or improved acceptance. Existing source/split and
semantic review gates remain required.

```bash
PYTHONPATH=src python -B -m pytest -q tests/test_prose_answer_scope.py tests/test_generation_grounding.py tests/test_scenario_plan.py tests/test_gold_provenance.py
PYTHONPATH=src python -B -m pytest -q
```
