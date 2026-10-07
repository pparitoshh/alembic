# Opt-in support review contract v2

`source-support-v2` replaces reviewer-invented evidence quotations with IDs selected
from a supplied registry. Each entry records exact content, origin and SHA-256; the
registry, source, transcript, schemas and visible policy are bound to the report.
Every target turn has its own chronological evidence allowlist. Only preceding,
replayed mock results and completed calls can support observations. Target answer
text is never registered as supporting evidence. Action workflows are outside this
read-only protocol and are rejected before a reviewer request.

Target-text coverage remains strict: ordered verbatim segments must cover every
non-whitespace character of every nonempty model-authored assistant turn. Omissions,
changed text, later-result citations and altered report bindings remain pending.
The reviewer still has to copy target segments correctly; the repair removes source
quotation reconstruction, not all possible output-contract mistakes.

Uncertainty is `{ "present": false, "reason": null }` or a true flag with a specific
nonempty reason. Strings such as historical `"none"` are not converted or approved.
The revised prompt distinguishes future intent, questions, stipulated scenarios,
hypothetical technical examples and asserted observations. Technical premises and
explanatory additions still need support, including within hypothetical examples.

Use the same explicit protocol in both stages:

```python
source = source_for_record(cfg, row)
report = request_review(reviewer, row, source,
                        protocol=cfg.verify.grounding_review_protocol)
```

Set `verify.grounding_review_protocol=source-support-v2` and point
`verify.grounding_reviews` to the new report file before `verify.run(cfg)`.
`require_grounding_review` and existing workflow-specific requirements still apply.
The default remains v1, whose request prompt, schema and historical outcomes stay
unchanged. Neither stored report labels nor hidden expected modes select the policy.
Do not mix protocols or rewrite original review files.

CPU regressions use the real `request_review`/`verify.run` paths, fake reviewers and
deterministic mocks. Captured checkpoint-06 reviews are exposed development fixtures:
the frozen outcomes remain 0 accepted, 0 rejected, 10 pending. All diagnostic fixtures
retain `purpose=diagnostic_only` and `training_eligible=false`. Synthetic v2 reviews
are **test fixtures**, not newly observed teacher judgments or accepted training data.

The observed seven citation failures have registry-selection regression coverage;
the two primary uncertainty failures have typed-field coverage; omitted answer text
still fails strict coverage. Intent/hypothesis examples and both unsupported addenda
have fake-reviewer regressions. The earlier model's three semantic mistakes are not
claimed fixed by CPU tests. A real but irrelevant citation can still accompany a
wrong reviewer verdict; a test makes this limitation explicit. Citation validity and
complete coverage do not establish entailment, task completion or model reliability.
Coordinator evidence review and model support judgments remain separate.

The remaining validation is a separately authorized, bounded **review-only** check
over the ten existing captured answers, retaining one raw v2 self-review per answer,
temperature 0, max_tokens 4096 and concurrency 1 with the same Qwen teacher. Reuse the
frozen source/transcripts/tool evidence; generate no conversations, change no answers,
and keep missing/malformed/uncertain reviews pending. Compare semantic judgments with
the preserved coordinator review and record disagreements. No such runtime check was
performed or submitted in this CPU-only checkpoint. These ten cases remain ineligible
for training regardless of future support-review success.
