# Fix 4, step 1: format audit (2026-10-28)

Training runs on Curiosity only. Leonardo has no budget. All data below was read on Curiosity in the team folder
`/home/eai-jjppv/cp` (read-only).

## Model lineage

ClusterPilot-4B 1.3 (`exp/kdall/model`) comes from this chain (from `KD_TRAIN.json` in each model folder):

| Model | Student (start) | Data |
|---|---|---|
| `kd1` | `models/r27` (round 27) | `exp/kd1/teacher` |
| `kd2` | `kd1` | `exp/kd2/teacher` |
| `kdall` (1.3) | `kd2` | union of the kd1, kd2 and kd3 teacher sets (136,452 answers) |

Round 27 is older than the tool rounds (round 38 simulated tools, round 49 real agent trajectories). The round 49
agent data is therefore not in 1.3.

## Format comparison

| Format | Source | Tool call text | Tool result |
|---|---|---|---|
| Round 49 training target | `research/agent_train/build_agent_data.py` (`call_text`) | `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>` | in prompt history |
| 1.3 chat template | `exp/kdall/model/chat_template.jinja` (Qwen3) | same `<tool_call>` block | `<tool_response>` in a user turn |
| Ollama Modelfile | `release/Modelfile` and local `hpc-assistant-vkkm-live` | same `<tool_call>` block | `<tool_response>` in a user turn |

The three formats agree. The tool-call format is not the cause of the faults.

## Tool rows in the 1.3 training data

| Check | Result |
|---|---|
| Prompts with a `tools` field (`exp/kdall/prompts.jsonl`) | 0 of 150,000 |
| Prompt build (`research/curiosity_b300/teacher_gen.py`) | system + one user turn, no `tools`, no tool results |
| Teacher answers with `tool_call` | 62 of 136,452; all are API documentation text (OpenAI `tool_calls`, test configs), not calls |
| Teacher answers with a `slurm_live` tool name | 1,557; generic words in scripts (`job_script` 1,213, `submit_job` 252, others 104), not calls |
| Targets with a tool call outside `<tool_call>` or in a code fence | 0 (there are no tool targets) |

## Conclusion

1. The 1.3 model has had no tool-use training. Its tool behaviour comes only from the Qwen3-4B base. The KD runs on
   single-turn HPC answers can also have weakened it. This explains the text tool calls and the early stops.
2. The current KD pipeline (`teacher_gen.py`, `train_kd.py`) supports only single-turn prompts without tools. Fix 4
   needs a prompt build with tool schemas and multi-turn history (tool calls and `<tool_response>` turns), rendered
   with the same chat template.
3. The round 49 builder (`build_agent_data.py`) already writes the correct target format. It can be the base for
   step 5, with the `mcp__hpc__` tool names that dsh uses.
