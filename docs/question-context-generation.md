# Self-contained question context

The question author sees a source chunk and may see a scenario brief. The
student's stored user turn contains only the generated question. These are
different contexts: source support for a numerical answer does not make a
question about an absent table answerable, and an exact source rewrite is not
a useful training target when the user never supplied the function or bounds.

Prompt version `source-grounded-v8-self-contained-resource-policy` explicitly asks
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
