# Bounded question decoding

Question requests keep the configured output-token limit; this repair does not increase it. `GeneratedQuestion` permits exactly one `question` field and still validates a minimum of 10 characters locally. Its versioned transport schema omits only that minimum: installed xgrammar 0.2.7 compiles a length-constrained string to a rule that cannot emit JSON escapes, including the newlines and quotes needed in code questions. The canonical semantic schema remains unchanged by transport conversion. Invalid short replies still fail validation. Answers and the judge schema/protocol are unchanged.

For a serving endpoint with verified vLLM support, a fresh run can explicitly set:

```yaml
generate:
  question_repetition_detection:
    min_pattern_size: 2
    max_pattern_size: 16
    min_count: 16
```

This optional setting applies only to question requests, retains other endpoint extras, and leaves answer decoding unchanged. The default is `null`. The three values are token-pattern bounds, not semantic-duplication or grounding criteria. A length or repetition termination cannot supply a question, even when its partial content happens to parse, and is not automatically retried by `chat_json`. Existing retries for completed invalid JSON remain bounded as before. The question-only change does not change judge handling of provider finish reasons.

The corresponding tested campaign launcher selects the installed xgrammar backend with `disable_any_whitespace: true`. This chooses the compiler's fixed JSON separator spacing and prevents arbitrary whitespace between fields; spaces and escaped newlines *inside* question strings remain valid. It is a server option, not an invented production-config key. A serving-version/schema probe must verify it before a future run.

Scenario generation already binds configuration, schemas, source/gold bytes and code hashes before constructing the teacher. A change requires a fresh run directory. No held-out selection or gold checks are relaxed. Raw unsuccessful attempts remain in campaign capture, while their absent answers remain absent and in the full plan denominator. A successful execution receipt for fully accounted incomplete questions is not dataset acceptance. Unknown/missing capture, incomplete eligible answers, runtime stops and binding failures stay failures.

Captured-text CPU probes distinguish repetition in raw response whitespace from repetition in the question value. Re-tokenization and prospective fixed-spacing fixtures are not new model outputs or historical acceptance decisions. It can stop a legitimate highly repetitive question and does not detect all degeneration or semantic defects. The new policy needs a measured future run; it supplies no evidence that old or future generated answers are correct. Historical artifacts are never repaired or relabeled.
