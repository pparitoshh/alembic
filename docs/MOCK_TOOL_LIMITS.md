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

Replay validation binds a trace to a particular mock version. Preserve the code
hash with generated data. Do not overwrite earlier raw results or reinterpret
them as fabricated solely because a later mock implementation changed.
