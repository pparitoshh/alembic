# Source-grounded generation prompts

`source-grounded-v2` keeps technical facts and command syntax tied to the assigned source, and runtime claims tied to current tool schemas and results. User constraints do not establish a diagnosis. Gold transcripts remain enabled and provenance-checked, but are explicitly examples of style and conversation format, not evidence for a new question.

Question generation receives complete tool argument schemas to distinguish a live lookup/action, missing required information, and a general question. Script tasks may request source-supported snippets or scoped fragments; they no longer always demand a complete sbatch/shell program. Safe discovery remains possible when it can resolve a missing target. Error results and partial checks must not be described as successful or comprehensive validation.

The regression tests capture requests through `generate.run` and the multi-turn `tool_trace` loop with a fake client. They establish that the instructions reach production prompts and held-out source markers remain absent. They do **not** establish that a model follows the instructions or that a generated answer is correct.

The opt-in job-status workflow now uses `source-grounded-v6-enforced-status-policy`
and `job-status-evidence-v2`. Its [deployment-visible policy and support review](JOB_STATUS_POLICY.md)
distinguish permitted discovery, model attempts, runtime intervention and unresolved
factual support. Generic generation retains its existing tool policy by default.

Before generation, inspect the actual task/persona/mode plan. Some source/task combinations are unsupported even with these prompts; choose a compatible configuration or defer them. The current planner does not yet represent a semantic unsupported-plan outcome. Do not force tool traces into unrelated source topics to hit a ratio.

Use a fresh run directory and freeze code, effective configuration, source/gold hashes, model revision and this prompt version. Existing deterministic IDs do not encode prompt or chunk revisions; resuming an old directory could reuse questions or answers made with older prompts. Preserve raw candidates and rejection reasons. Apply grounding and mode review after generation before training acceptance; automated schema/syntax checks alone are insufficient.
