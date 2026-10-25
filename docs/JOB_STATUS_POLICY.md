# Job-status policy and factual support

`job-status-evidence-v2` adds application enforcement to the production `tool_trace`
loop and a separate support-review gate in `verify.run`. This is a bounded,
read-only simulator workflow, not a general authorization parser or a live Slurm
agent. Existing untagged workflows and the historical v1 results keep their
original policy. No historical case is relabelled by this change.

Use the typed configuration fields:

```yaml
generate:
  job_status_discovery: clarify_first
verify:
  require_grounding_review: true
  grounding_reviews: /path/to/this-run/support_reviews.jsonl
```

These are additions to a complete project config. `job_status_discovery: null`
retains generic generation. `allow_readonly` explicitly permits queue discovery
before clarification. Both policies are rendered in the teacher's system prompt
and stored on each record; expected modes, case IDs and reference answers never
select runtime behavior. The fixed-case harness requires a policy and grounding
review, and verifies that the case plan declares that same policy.

With `clarify_first`, an unidentified job requires clarification before lookup.
Explicit queue/resource-list requests remain permitted without a job ID. With
`allow_readonly`, the model may list candidate jobs and then ask the user to
select one. A returned candidate, even the sole candidate, does not establish
that it belongs to a named script. Inspecting the sole result is permitted only
when the user explicitly requested that selection. Multiple identified targets
require selection in this single-job workflow. Explicit instructions prohibiting
lookup take precedence. The conservative phrase matcher can reject unusual
valid phrasing; clarification is its safe fallback, not proof of general intent
understanding.

All job-specific calls require IDs supplied in user context or actual prior
tool results, plus a permitted selection. Tool errors and log strings are not
ID evidence. Each model-authored parallel batch is validated against evidence
available before that batch. A violation blocks the entire batch: one proposed
call cannot authorize another simultaneous call. Mutating tools are blocked in
this workflow. Other generic tool workflows remain available when this policy
is not enabled.

The raw assistant attempt stays in the transcript. A blocked attempt receives
an explicit tool error and an application-authored clarification with
`origin: runtime_guard`. `runtime_audit` records each attempted call, execution,
block reason, the model's final answer (if any), delivered response and response
origin. Replay checks the audit against the same policy and mock state. Safe
enforcement does not make the model compliant; such attempts fail verification.
The training adapter rejects guard-authored responses and diagnostic-only rows.

## Grounding beyond command names

Shell/flag and schema checks remain useful but do not establish prose factuality.
V2 records require a separate `source-support-v1` review even if the optional
config flag is omitted. The gate resolves the assigned training source and its
recorded hash, checks schemas, and binds the review to the transcript, policy,
source, schema and review-prompt version. Held-out sources are forbidden. The
review input has no gold answers, expected mode, case ID or reference verdict.

`distillkit.support.request_review(reviewer, row, source)` is an explicit optional
model request; `verify.run` never starts inference. A caller must preserve its
returned report as one JSONL row in `grounding_reviews`. The report retains raw
response, parsed segments, reviewer identity, time, prompt hash, packet hash,
citations, explanations and uncertainty. The existing `Teacher` interface can
be used for an authorized reviewer; no new model or judge protocol is selected
by this code. Self-review must be labelled as such, not independent review.

The reviewer must cover every assistant text turn with verbatim claim segments.
General claims require source/schema support; current observations require
preceding tool results. User text can stipulate requirements but cannot prove a
diagnosis. A result from a later turn cannot support an earlier claim. Supported
paraphrases and hypothetical examples are allowed. An unsupported hypothetical
cause is still unsupported, but is not necessarily an asserted cause of the
user's actual failure or a false statement.

Tool-bearing records cannot bypass replay by using a prose/missing mode label.
Review evidence pairs each completed call's name and arguments with its result,
available only after that result. This protocol reviews assistant text and the
bounded read-only diagnostic context. It does not approve arbitrary scripts or
other artifacts inside action arguments: support-required rows using other
tools stay pending for separate argument-content review. Pending and rejected
questions do not displace a later supported copy during deduplication.

| Outcome | Verification destination |
|---|---|
| Structurally valid, support review says supported, bindings/citations/coverage valid | `verified.jsonl` |
| Valid review identifies an unsupported claim | `rejected.jsonl` with `unsupported_claim` |
| Missing, malformed, duplicate, stale, incomplete or uncertain support evidence | `pending_review.jsonl` |
| Policy, trace, command or other hard failure | `rejected.jsonl` with the relevant reason |

These stages do not establish training eligibility by themselves. Citation
existence and coverage are deterministic checks, not semantic entailment. Model
review remains fallible. Faithfully describing the wrong job may be supported
yet fail the user's request; assess task completion separately. Keep provenance,
holdout, deduplication and task-completion review before training acceptance.

Tests exercise production conversation, mock replay, fixed-case orchestration
and verification with fake teachers/reviewers and real dependencies. They do
not demonstrate changed behavior of a real teacher. A fresh authorized trial
must separately report model compliance, guard enforcement, factual support,
task completion and pending evidence. Preserve both prior 10/12 teacher results.
