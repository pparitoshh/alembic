# Training suitability and Slurm value checks

Training verification now checks literal Slurm time and distribution values as
well as option names. Unknown or dynamic values remain outside these conservative
checks; a parser pass is not proof that a job can run on a particular cluster.
The evaluation checker keeps its existing default behavior. Callers can explicitly
enable the extra checks with `check_option_values=True`.

Answer generation now asks for the complete supported answer, including relevant
commands, prerequisites and checks, without a fixed sentence limit. Site-specific
settings must stay attributed to their cluster. This prompt change does not
establish a student-quality improvement.

To inspect a JSONL dataset without loading a model:

```sh
python -m distillkit.data_profile records.jsonl --output profile.json
```

Keyword counts are discovery aids, not relevance judgments. The optional
`train.data_requirements` config adds an exact-record review and coverage check
after existing provenance and held-out-source checks, before model loading:

```yaml
train:
  data_requirements:
    domain_review:
      path: reviews/domain.json
      sha256: <SHA-256 of the review file>
    min_tool_fraction: 0.25
    max_tool_fraction: 0.35
    required_modes: [call, ask, none]
    required_tasks: [concept, howto, script, debug]
```

The review JSON uses version `training-domain-review-v1`, the profile's
`records_sha256`, decision `approved-for-hpc-training`, every selected ID in
`approved_record_ids`, a `review_method`, and an `evidence` object containing
`path` and `sha256`. Evidence paths are relative to the review file. The gate
checks hashes, coverage and structure; it trusts the supplied reviewer and does
not certify semantic correctness. Source verification, decontamination and tool
validation remain required. Calls, clarification requests and deliberate no-call
decisions all count toward the tool fraction.

Omitting `data_requirements` preserves existing admission behavior and historical
training fingerprints. Evaluation questions, references and scoring fixtures are
unchanged; the release manifest records the config binding transition and its
previous hash.
