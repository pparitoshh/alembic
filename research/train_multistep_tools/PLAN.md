# Training fix 4: multi-step tool use

Status: plan only (2026-10-28). No data and no training yet.

## Problem

The dsh mentor tests show that the HPC assistant (`hpc-assistant-vkkm-live`, Qwen3-4B) does not use its tools well.

| Metric | Leonardo tasks (14) | Curiosity tasks (16) |
|---|---|---|
| Tasks with real tool calls | 3 | 5 |
| Real tool calls | 4 | 5 |
| Mean recall of expected tools | 0.02 | 0.07 |
| Tasks with tool calls written as text | 7 | 9 |
| Answers that stop at "I will now check" | 3 | 2 |

Scores come from `research/dsh_harness/mentor_ui.py score`.

The model shows three faults:

1. It writes a tool call as text (a JSON block or a tool name in a bash block). The harness then runs nothing.
2. It stops after one tool call. It does not continue with a second tool.
3. It invents values (partitions, hosts, limits) when it has no tool result.

The tool-call repair proxy (`research/dsh_harness/toolcall_proxy.py`) and the cluster facts card in the dsh profile reduce faults 1 and 3 at inference. Only training can fix the cause.

## Goal

| Target | Now (Curiosity) | Goal |
|---|---|---|
| Tasks with real tool calls (no proxy) | 5/16 | 12/16 or more |
| Tasks with tool calls written as text | 9/16 | 2/16 or less |
| Mean recall of expected tools | 0.07 | 0.5 or more |
| Reviewed leaderboard mean | 1.3 baseline | not lower than −0.015 |

## Rules

- The mentor use cases (`data/mentor_usecases/`) are held-out test data. Never use them for tasks, trajectories, gold examples or retrieval.
- Check every new task against the mentor use cases for overlap. Remove a task if its word 5-gram overlap with a mentor request is more than 0.3.
- Real cluster jobs only with the user's approval. Action tools (`submit_job` and the other five) get `test_only` or simulated results during collection.
- New runs use bf16 base weights (no 4-bit training). Reports show 3-seed means only.

## Steps

1. **Format audit.** Compare three formats: the `<tool_call>` target text in the round 49 data (`research/agent_train/build_agent_data.py`), the chat template of the released model, and the Ollama Modelfile template. Count how many training targets contain a tool name outside a `<tool_call>` block or in a code fence. Output: a short table in this folder.
2. **Task bank.** Write new user requests for the 18 tools of `distillkit.slurm_live`, for Curiosity and for Leonardo. At least 50% of the tasks need two or more tool calls (for example `gpu_availability` → `partition_info` → answer, or `job_status` → `read_job_log` → answer). Add tasks where the correct answer needs no tool, and tasks where the model must ask for a missing job ID. Run the overlap check against the mentor use cases.
3. **Trajectory collection.** A teacher (Qwen3-235B-A22B FP8 on Curiosity, as in the distillation loop) runs each task through the real MCP server (`research/dsh_harness/mcp_server.py`) in read-only mode, with real tool results. Use `research/agent_train/collect.py` as the base.
4. **Verification.** Keep a trajectory only if all of these are true:
   - every tool call parses and its arguments pass the tool schema;
   - no tool name appears in the answer text;
   - every job ID, partition and number in the final answer comes from the user turn or a tool result;
   - the trajectory has at most 6 tool calls.
5. **Data build.** Make one example per assistant turn, as `build_agent_data.py` does. Use the tool names and schemas that dsh gives at inference (`mcp__hpc__` prefix). Add replay rows from the 1.3 data so that the HPC answers stay good.
6. **Training.** Start from ClusterPilot-4B 1.3 (`kdall`). Do full-parameter distillation with `research/curiosity_b300/train_kd.py` (bf16) on the 8 B300 GPUs. Run 3 seeds.
7. **Evaluation.**
   - Tool format: `research/dsh_harness/test_toolcalls.py`.
   - Leaderboard: `research/curiosity_b300/eval.sh`, compared with 1.3 in the same run.
   - Mentor tasks in dsh: `mentor_ui.py run` on `data/mentor_usecases/curiosity/usecases.jsonl`, one time without the proxy (training effect alone) and one time with the proxy.
8. **Release.** If all goals pass: GGUF build, Ollama Modelfile with the same tool template, the next version number.

## Open questions

- Teacher choice: Qwen3-235B FP8 (known) or gpt-oss-120b (better tool calls?). A small probe on 50 tasks can decide.
- GPU time: the team limit on Curiosity is 8 GPUs. Collection and training cannot run at the same time as other team jobs.
