# Batch-local source resolution

`support.source_for_record(cfg, row)` retains its existing uncached behavior.
For a batch, `support.source_for_records(cfg, rows)` captures source and registry
bytes once, validates the same admission rules, checks every row's original
metadata and source hash, then rehashes the snapshot before returning.

Verification and direct registered-source training admission use the same
`SourceResolver`. Multi-origin training creates one resolver per original
configuration, keeping original registries, source hashes and legacy admission
separate. All existing cross-origin checks and final bound-file checks remain.
No record, prompt, review protocol, split, or model setting changes.

Callers that need to interleave source resolution and other work can construct
`SourceResolver(cfg)`, call `source_for_record(cfg, row, resolver=resolver)`, and
call `resolver.assert_unchanged()` immediately before accepting or persisting
results. This last call is required; resolving one row from a snapshot alone is
not a fresh filesystem check. The convenience batch API performs it for callers.
The verifier calls it before writing any output; training admission calls it
before returning rows or persisting admission evidence.

The snapshot binds selected source names and resolved paths, original file bytes,
registry bytes and the complete serialized effective Config. Config has no
backing-file attribute; an owner with a backing configuration may also pass
`config_path=` to bind and recheck that file's bytes. Multi-origin admission does
this for every original configuration. Source membership, file-content, registry,
or configuration changes cause a failure. Checks do not depend on mtimes and do
not silently reload changed data. No process-global or cross-run cache exists.
Full and chunk source strings preserve the existing text decoding and newline
normalization. Returned chunk/metadata collections are copies.

This is an optimization of immutable batch inputs, not a claim that a mutable
filesystem is atomically locked. Changes that occur and are restored entirely
between checks are not observable. No on-disk write follows a detected mutation;
callers remain responsible for a private/frozen run directory. Semantic support,
source permission and model quality are unchanged.
