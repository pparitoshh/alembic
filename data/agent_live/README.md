# Agent live task set

## What this directory holds

This directory holds `tasks.jsonl`. The file lists prompts for a live agent run of `hpc-assistant --agent` on a real Slurm cluster (Leonardo, account `EUHPC_D30_031`). One JSON object sits on each line:

- `id`: a short unique name for the task.
- `prompt`: the user question, in natural wording.
- `expected_tool`: the tool the agent should use.
- `expected_args`: only the key arguments; other arguments (like a full script) are not listed.
- `is_action`: `true` for the cluster-changing tools (`submit_job`, `cancel_job`, `update_job`, `hold_job`, `release_job`, `requeue_job`), else `false`.

Each tool in `distillkit.slurm_live.LIVE_SCHEMAS` has at least two prompts. `run_readonly_command` prompts ask questions that no fixed tool answers.

## This is not mentor held-out data

This task set is working material for live smoke tests in the hackathon (plan `agent_harness_plan.md`, step 8). It is NOT the hold-out task set of the mentor evaluation and must not replace it. Do not use these prompts as a readiness signal for the agent rounds.

## Placeholders

Three placeholders stand for real job ids. Fill them before the run:

- `<RUNNING_JOB>`: the id of one of the user's running jobs (for example from `squeue --me -t RUNNING`).
- `<PENDING_JOB>`: the id of one pending (or previously held) job of the user.
- `<FINISHED_JOB>`: the id of a completed or failed job of the user, from `sacct`.

Replace each placeholder with a plain job id string (for example `60340010`) before you send the prompt. Keep the id of the same job across tasks that refer to the same run.
