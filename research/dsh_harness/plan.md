# dsh harness

## Goal

Run the HPC assistant student model behind a local harness with a web UI. Register our 16 Slurm tools. Keep telemetry local for debugging. First option: DeepSeek Harness (dsh). Second option: Hermes Agent (Nous Research).

## Background

dsh is open source (MIT) and in Developer Preview. Its settings file is `~/.dsh/settings.yaml`. dsh accepts any OpenAI-compatible endpoint as a custom provider through the llm-pi-ai plugin. Tools in dsh are plugins. A plugin is a tool form plus a `dsh-plugin.naming.json` manifest. Telemetry in dsh is built on the session log. The command `dsh --profile web --dump-config` prints the plugin tree.

Qwen3 uses the Hermes tool-call text format `<tool_call>{...}</tool_call>`. Hermes Agent is a different program. It needs the server to return structured `tool_calls`.

The Hugging Face model `hf.co/VaibhavKKM/hpc-assistant-qwen3-4b` has a thinking-aware Ollama template. That template puts tool calls in the thinking field. The local model `hpc-assistant-vkkm-live` uses the same weights with the x02 template. It works: 8 of 8 questions gave a tool call, and 6 of 8 first tools were correct.

## Steps

1. Get llama-server. It is not installed. The host has no compiler. Docker exists. Ask the user before any download.
2. Run llama-server with the Q4_K_M GGUF, flags `--jinja -c 16384`. Reuse the GGUF blob already in the Ollama store so nothing is downloaded. Blob: `sha256-f32f2cddcedb70c150232283220e2c83cfd0fbda89a73b80016fbb08ded56208` under `/usr/share/ollama/.ollama/models/blobs/`.
3. Run `research/dsh_harness/test_toolcalls.py` against `http://127.0.0.1:8080/v1`. Check that replies contain real `tool_calls`.
4. Install dsh (pin the version). Run `dsh --profile web --dump-config`.
5. Read the `docs/` of the official repo `deepseek-ai/deepseek-harness` for tool registration and telemetry keys.
6. Add a provider block in `~/.dsh/settings.yaml` for the local endpoint.
7. Write a dsh plugin that registers the 16 tools from `src/distillkit/slurm_live.py` (use `openai_tools()`). Actions that change jobs must ask for confirmation, same as the agent.
8. Turn on telemetry and check that it stays local.
9. Compare with Hermes Agent if dsh does not fit.

## Constraints

- Use temperature 0 for tests.
- Send read-only questions only on Leonardo.
- Never commit a password.
- The Leonardo password is in `.env` as `LEONARDO_PASSWORD`. Never print or copy it.

## Risks

- The template inside the GGUF may bring back the thinking problem. Test this in step 3.
- dsh is in Developer Preview and can break.
- Telemetry may send data outside the host. Check before you enable it.

## Open questions

- What is the exact dsh tool registration API?
- What are the telemetry settings keys?
- How do we confirm actions in the dsh web UI?

## Status

Steps 1 to 9 not started.

Host check (Task C), 2026-10-27: `which llama-server dsh` found neither program (no output). `docker images | head` shows these local images (no pulls needed later, no llama-server image present):

```
ai-chef-api:latest           2330b1773306        759MB
ai-chef-streamlit:latest     c859c1016793        673MB
grafana/grafana-oss:latest   5dad0df181cb        1.47GB
pgvector/pgvector:pg16       a36250871de0        621MB
texlive/texlive:latest       8.76GB
ubuntu:24.04                 119MB
```
