# Pairwise judge scoring

Protocol `paired-orders-v2-invalid-incomplete` keeps the existing verdict-only prompt,
schema, decoding and JSON-retry behavior. Both judges use the same protocol.

Each question is judged twice with answer order reversed. A valid judgment scores a
win as 1, a genuine tie as 0.5, and a loss as 0. The two scores are averaged, then
questions receive equal weight. Order disagreement is retained; reversal does not
guarantee absence of position bias.

An unparsed verdict is stored as JSON `null`, with its raw response preserved. If
any verdict in a comparison is invalid, the aggregate win rate is `null` and the
comparison is marked `incomplete`. Counts of expected verdicts, invalid verdicts
and complete pairs remain available. There is no headline score over a silently
selected valid subset. An empty comparison is also incomplete.

Across training seeds, a missing/null win rate (including a prepared seed directory
with no summary), mixed judge protocols, or mixed judge identities suppresses the
combined win-rate mean. The report retains expected/available counts and protocol
identities. Fully legacy results remain readable as `legacy-unversioned`; they
cannot be silently combined with v2. This check does not establish that the other
frozen inputs match; manifests and fresh comparison directories are still required.

Earlier evaluation code replaced invalid verdicts with ties. Results from that
policy must not be mixed with v2 scores. Re-score the same frozen answer sets
uniformly when changing protocols; preserve prior outputs. Judge calibration
already counted invalid judgments as misses and is unchanged.

Freeze the dataset hashes, answer sets, model identities, prompts, decoding and
this protocol version before a comparison. Use fresh output directories when
these change: the existing answer cache does not bind all those inputs. Calibration
success is not student accuracy, and agreement between judges is not proof of
correctness. Inspect capability-level outcomes alongside aggregate comparisons.
