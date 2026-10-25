# Opt-in support review contract v3

`source-support-v3` supplies deterministic target units instead of asking a
reviewer to copy and segment the answer. Each unit contains an exact text span,
Python character offsets, its text SHA-256 and an ID bound to the span, turn and
unit algorithm version. The original assistant turn remains in the packet for
context. Concatenating its units reproduces every character and UTF-8 byte;
whitespace and code are not rewritten.

The versioned, conservative splitter uses sentence punctuation, paragraph/list
boundaries and whole fenced code blocks. Decimal numbers, common abbreviations
and balanced inline code remain intact. Unclosed code fences remain one block.
This is not semantic claim extraction: a unit can contain several claims or need
the rest of its turn for interpretation. Every factual addition within a unit
must be reviewed, including a whole code block's technical content. No unit ID
encodes a correct verdict or case-specific rule.

The evidence registry, deterministic tool replay and per-turn allowlists come
directly from v2. Assistant targets are not evidence. Only preceding, replayed
mock observations can support live facts; later or unexecuted results cannot
justify an earlier unit. Action workflows remain outside this read-only evidence
contract. This change does not add tool execution or broaden those permissions.

## Reviewer response and strict verification

The new response schema is:

```json
{
  "units": [
    {
      "unit_id": "select-an-exact-supplied-unit-id",
      "verdict": "supported",
      "evidence_ids": ["select-an-available-registry-id"],
      "explanation": "Explain how the evidence supports every factual claim in this unit."
    }
  ],
  "uncertainty": {"present": false, "reason": null}
}
```

Every supplied unit must appear exactly once. Response order does not matter;
unknown, repeated or missing units remain pending. The reviewer does not return
target text, `kind` or a second support label. Its single verdict is one of:

- `supported`: every factual claim is supported by available evidence, with at
  least one citation. Faithful paraphrases and direct applications of stated
  rules to explicit scenario values are allowed without verbatim matching.
- `unsupported`: some factual claim, technical premise or addition lacks support
  or contradicts the evidence. This does not assert that the claim is false.
- `uncertain`: a specific support or interpretation question remains unresolved.
- `nonfactual`: a pure question, future intent, offer or formatting with no factual
  technical premise. A question or plan with such a premise still needs support.

The prompt distinguishes future intent from claims of completed actions, and
source-supported hypotheses from unsupported causal explanations. Allocation
counts or job names alone do not establish configuration/scheduling history;
status and exit-code limitations do not supply an undocumented cause taxonomy.
There is no deterministic heuristic that proves the reviewer's semantic labels.

The report binds the protocol, prompt, unit algorithm, target units, packet and
registry. Raw responses remain preserved. Malformed JSON, duplicate JSON keys,
extra fields, invalid citations, unavailable future results and altered bindings
remain pending; the validator never rewrites a response to make it valid.
Any `uncertain` unit or explicit global uncertainty keeps the record pending,
including a record that also has an unsupported unit. With complete valid
coverage and no uncertainty, an unsupported unit rejects the record; otherwise
supported/nonfactual units pass this support gate. These are support outcomes,
separate from tool policy, task completion and other acceptance checks.

## Explicit production configuration

Set both stages to the same protocol:

```yaml
verify:
  require_grounding_review: true
  grounding_review_protocol: source-support-v3
  grounding_reviews: path/to/new-v3-reports.jsonl
```

```python
source = source_for_record(cfg, row)
report = request_review(reviewer, row, source,
                        protocol=cfg.verify.grounding_review_protocol)
```

Use the existing `verify.run(cfg)` for acceptance. It calls the configured
production support dispatcher. The default stays v1. V1/v2 request prompts,
schemas, implementation and historical report bytes are unchanged; neither old
reports nor their labels are converted into v3. Request a new review explicitly
and retain the old report file.

## Evidence and limitations

Captured Gemma v2 reviews exhibited duplicated copied segments and contradictory
labels for future intent. Some direct source applications were also rejected for
not matching the source's wording. V3 removes copied-target coverage and the
`kind` × `support` combination, while clarifying semantic instructions. These
contract changes alone do not establish better model judgment.

`tests/fixtures/support_gemma_v2` preserves ten original diagnostic report lines
byte for byte, selected from the eighteen captured Gemma reviews. The fixture
manifest records the origin, hashes and selection; it reuses the existing
checkpoint-06 source/transcripts. These are exposed development regressions,
never training data or blind final-test material. Frozen v1 outcomes are ten
pending; those ten captured Gemma v2 outcomes are nine rejected and one pending.
New v3 test responses are explicitly synthetic, not model-generated reviews.

CPU tests exercise actual `request_review` and `verify.run` dispatch with fake
reviewers: exact coverage, code/abbreviation handling, supported prose and rule
application, pure intent versus embedded premises, both exposed unsupported
addenda, missing/future evidence, strict uncertainty, raw-response integrity and
unchanged v1/v2 outcomes. Deliberately wrong supported and nonfactual labels show
that valid unit IDs/citations do not prove entailment or factuality. Passing
these tests is not a real inference run or evidence of semantic improvement.

## Bounded v3 runtime result

The 2026-10-07 review-only run `support-units-002` (job `59659426`) used
`google/gemma-4-26B-A4B-it` against frozen code `e9eec23`. It reviewed the same ten
historical answers plus eight new synthetic diagnostic controls, once each, at
temperature 0, a 4,096-token response limit and concurrency 1. It generated no
new assistant answers and contributes **zero training-eligible records**.

| Runtime outcome | Records |
| --- | ---: |
| Responses received | 18 |
| Schema-valid reviews | 13 |
| Complete valid unit coverage and citations | 12 |
| Kept by the configured verifier | 10 |
| Rejected | 2 |
| Pending | 6 |

Independent model-assisted source review found two false accepts among the
twelve valid contracts. Both were historical answers: one added unsupported
configuration/scheduling history from a zero GPU count; the other added a cause
taxonomy and a memory-diagnosis rule absent from its source evidence. Valid
citations and complete unit coverage did not make those additions supported.

Five responses exhausted the token limit while emitting trailing whitespace
after their units array, omitting the uncertainty field and final object. Their
visible prefixes did not repeat unit IDs. The sixth pending review dropped one
digit from a target-unit ID. The strict verifier preserved all six as pending;
no truncated response or altered ID was repaired. The historical ten yielded
five semantic agreements, two false accepts and three contract failures. The
fresh eight yielded five agreements and three contract failures. These small,
exposed diagnostic sets are not blind final tests or evidence of student-model
improvement; the independent review is also fallible and was not human review.

V3 remains experimental and explicitly opt-in; the default remains v1.
**Gemma v3 must not be the sole basis for training eligibility.** Any pilot using
these reviews also needs separate per-record source/evidence review and the
existing hard validation checks. Keep contract failures, semantic disagreements
and their denominators separate. Preserve all v1/v2/v3 raw reports and historical
outcomes; do not relabel them or tune labels to improve counts. This limitation
does not require removing the experimental protocol or a mandatory human-review
stage.
