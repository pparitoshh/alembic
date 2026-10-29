# Round 2, step 0: mentor UI baselines at temperature 0

Date: 2026-10-29. One run per model and mode (16 Curiosity mentor cases per run).

Models: `hpc-assistant-fix4-t0` (from `hpc-assistant-fix4:latest`, round 1) and
`hpc-assistant-vkkm-live-t0` (from `hpc-assistant-vkkm-live:latest`, round 1.3).
Both Modelfiles set `PARAMETER temperature 0`. Runs stay in `~/.claude/handovers/r2_baseline_<model>_<mode>_20261029`.

## Totals

| Value | fix4-t0 direct | fix4-t0 proxy | vkkm-live-t0 direct | vkkm-live-t0 proxy |
|---|---|---|---|---|
| cases | 16 | 16 | 16 | 16 |
| cases_with_real_tools | 10 | 12 | 3 | 8 |
| real_tool_calls | 32 | 37 | 3 | 11 |
| mean_recall | 0.161 | 0.380 | 0.016 | 0.130 |
| fake_tool_call_cases | 0 | 1 | 1 | 2 |
| stopped_early_cases | 0 | 0 | 0 | 1 |
| tool_error_cases | 3 | 2 | 0 | 0 |
| empty_answer_cases | 1 | 0 | 0 | 0 |

Notes:

- fix4-t0 has more real tool calls than the round 1 temperature 0.7 baseline (9 direct, 8 proxy).
- The proxy raises recall for both models. The direct and proxy numbers still differ much.
- vkkm-live-t0 (1.3) rarely calls a tool in direct mode: 3 of 16 cases.

## Failure types: cases with no tool call, fix4-t0 direct

6 of 16 cases have no tool call. Each case answers from memory:

- curiosity-002: answers from memory
- curiosity-003: answers from memory
- curiosity-005: answers from memory
- curiosity-007: answers from memory
- curiosity-009: answers from memory
- curiosity-014: answers from memory

No case asks the user. No case stops after a plan. No case belongs to another type.
