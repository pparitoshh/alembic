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

## Round 2 model (v3), temperature 0

Date: 2026-10-29. One run per mode (16 Curiosity mentor cases per run). Model
`hpc-assistant-fix4v3-t0:latest` (from the round 2 v3 weights, Modelfile sets `PARAMETER temperature 0`).
Runs in `~/.claude/handovers/r2_v3_direct_20261029` and `~/.claude/handovers/r2_v3_proxy_20261029`.

| Value | v3 direct | v3 proxy |
|---|---|---|
| cases | 16 | 16 |
| cases_with_real_tools | 10 | 11 |
| real_tool_calls | 29 | 35 |
| mean_recall | 0.208 | 0.307 |
| fake_tool_call_cases | 0 | 0 |
| stopped_early_cases | 0 | 0 |
| tool_error_cases | 3 | 2 |
| empty_answer_cases | 1 | 3 |

Notes:

- Direct: v3 has the same number of cases with tools as fix4-t0 (10) and a higher recall (0.208 against 0.161).
- Proxy: v3 has one case with tools less than fix4-t0 (11 against 12) and a lower recall (0.307 against 0.380).
- The proxy raises recall again (0.208 against 0.307), but the gain is smaller than for fix4-t0 (0.161 to 0.380).
- No case writes a tool call as text in either mode (fake_tool_call_cases 0). No case stops early.

## Failure types: cases with no tool call, v3 direct

6 of 16 cases have no tool call. Type per case:

- curiosity-002: answers from memory
- curiosity-003: answers from memory
- curiosity-005: answers from memory
- curiosity-008: answers from memory
- curiosity-011: stops early (one sentence announces the work; the answer has no steps, no plan and no tool call; the scorer flags an empty answer)
- curiosity-015: answers from memory

Three case ids (002, 003, 005) also have no tool call in the fix4-t0 direct baseline; 007, 009 and 014 now call tools, and 008, 011 and 015 now do not. No case asks the user. No case belongs to another type.

## Release rule (PLAN_round2.md)

| Rule | Round 1 | Round 2 | Result |
|---|---|---|---|
| Direct recall higher than round 1 | 0.161 | 0.208 | passed |
| Cases with tool calls as text: 2 or less | 0 | 0 | passed |
| Leaderboard mean not lower than round 1 | 0.850 | 0.858 | passed |

Round 2 passes the release rule. The proxy recall is lower than round 1 (0.307 against 0.380), and the goals of plan step 7 (recall 0.5, 12 cases with tools) are still not reached.
