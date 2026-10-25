# Deterministic mock limits

The tool implementations are synthetic fixtures, not a live scheduler. GPU
availability counts must be internally consistent: fully idle nodes cannot
exceed the partition's node count or account for more GPUs than are free.
`submit_job(test_only=true)` reports the requested partition, with this mock's
default used only when no partition is specified. It does not invent a CPU
allocation that was not resolved from the script.

The returned `validation_scope` lists the implemented checks: recognized flag
names, Bash syntax and partition name. It is not full Slurm validation. In
particular, resource combinations, time limits, account permissions, actual
capacity and policy constraints require separate checks. A synthetic future
start time is not a real scheduling prediction.

Generation and replay now enter a fresh `mock_session` per conversation. Successful
mock submissions create pending records; cancellations update later status/queue
observations. Test-only submissions do not change state. No clock, dispatch,
capacity reservation, execution or time-limit validation is simulated. Submitted
scripts are retained verbatim and unresolved resource fields remain unknown.
Concurrent conversations cannot observe each other's actions. One-shot evaluation
continues to use independent initial fixtures; its scoring policy is unchanged.

Logs derive their GPU count/type and node name from the job fixture. CPU-only
accounting explicitly reports zero GPUs. Question generation receives the full
job/partition premise and stores it for review; an instruction is not proof that
the teacher obeyed it. `generate.tool_doc_ids` optionally limits tool modes to
compatible **training** documents; omitted means the previous planner behavior.
The persona/task grid and question IDs are unchanged.

Replay validation binds a trace to a particular mock version. Preserve the code
hash with generated data. Do not overwrite earlier raw results or reinterpret
them as fabricated solely because a later mock implementation changed.
New question/answer rows carry prompt and mock version labels. Unknown explicit
mock versions fail replay. Older rows without labels require their archived code
for a trustworthy historical audit; absence of a label is not proof of compatibility.
