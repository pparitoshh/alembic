# dsh harness

## Goal

Run the HPC assistant student model behind a local harness with a web UI. Register our 16 Slurm tools. Keep telemetry local for debugging. First option: DeepSeek Harness (dsh). Second option: Hermes Agent (Nous Research).

## Background

dsh is open source (MIT) and in Developer Preview. Its settings file is `~/.dsh/settings.yaml`. dsh accepts any OpenAI-compatible endpoint as a custom provider through the llm-pi-ai plugin. Tools in dsh are plugins. A plugin is a tool form plus a `dsh-plugin.naming.json` manifest. Telemetry in dsh is built on the session log. The command `dsh --profile web --dump-config` prints the plugin tree.

Qwen3 uses the Hermes tool-call text format `<tool_call>{...}</tool_call>`. Hermes Agent is a different program. It needs the server to return structured `tool_calls`.

The Hugging Face model `hf.co/VaibhavKKM/hpc-assistant-qwen3-4b` has a thinking-aware Ollama template. That template puts tool calls in the thinking field. The local model `hpc-assistant-vkkm-live` uses the same weights with the x02 template. It works: 8 of 8 questions gave a tool call, and 6 of 8 first tools were correct.

## Steps

1. SKIPPED by user decision: get llama-server.
2. SKIPPED by user decision: run llama-server with the GGUF. Note: llama-server remains an option when the user is on Wi-Fi (it needs a download or a build).
3. Run `research/dsh_harness/test_toolcalls.py` against the Ollama endpoint `http://127.0.0.1:11434/v1` with model `hpc-assistant-vkkm-live:latest`. Check that replies contain real `tool_calls`.
4. Install dsh (pin the version). Run `dsh --profile web --dump-config`.
5. Read the `docs/` of the official repo `deepseek-ai/deepseek-harness` for tool registration and telemetry keys.
6. Add a provider block in `~/.dsh/settings.yaml` for the endpoint `http://127.0.0.1:11434/v1`.
7. Write a dsh plugin that registers the 16 tools from `src/distillkit/slurm_live.py` (use `openai_tools()`). Actions that change jobs must ask for confirmation, same as the agent.
8. Turn on telemetry and check that it stays local.
9. Compare with Hermes Agent if dsh does not fit.

## Constraints

- Use temperature 0 for tests.
- Send read-only questions only on Leonardo.
- Never commit a password.
- The Leonardo password is in `.env` as `LEONARDO_PASSWORD`. Never print or copy it.

## Risks

- The model `hpc-assistant-vkkm-live` uses the x02 template. Do not switch to the `hf.co/...` model. Its thinking template breaks tool calls.
- dsh is in Developer Preview and can break.
- Telemetry may send data outside the host. Check before you enable it.

## dsh facts (from web research, nothing installed)

Sources: README of github.com/deepseek-ai/deepseek-harness, the npm registry entry, and third-party pages. The `docs/` URLs `architecture.md` and `docs/` returned 404. I did not read `docs/user/guide/index.md` or `SAFETY.md`.

| Item | Result |
| --- | --- |
| Install command (npm) | `npx @deepseek-ai/dsh web` (needs Node.js). The Web UI opens at `http://127.0.0.1:3080`. Add `--no-open` to skip the browser. |
| Install from source | `git clone https://github.com/deepseek-ai/deepseek-harness.git`, `pnpm install`, `pnpm run build`, `pnpm dsh web` |
| Pinned version | npm dist-tag latest is `0.2.1-rc.2`. The GitHub Releases section is empty. Pin command: `npx @deepseek-ai/dsh@0.2.1-rc.2 web`. |
| Download size | Package `@deepseek-ai/dsh` alone: 75,972 bytes unpacked, 20 files. It has 86 direct dependencies. Total download size: not found. |
| Plugin manifest | A third-party page says `dsh-plugin.naming.json` sits at the plugin repository root. Field list: not found. |
| Tool form for a plugin | not found |
| Telemetry settings keys | not found |
| Telemetry data destination (local or remote) | not found |
| Settings file | `~/.dsh/settings.yaml` (or `$DSH_HOME`). Secrets go in `.credentials.yaml`. Profiles are in `$DSH_HOME/profiles/<name>`. |

Other notes: dsh is a Developer Preview. The README warns of compatibility-breaking changes and asks you to read `SAFETY.md`.

## Open questions

- What is the exact dsh tool registration API? (not found)
- What are the telemetry settings keys? (not found)
- Where does telemetry data go? (not found)
- How do we confirm actions in the dsh web UI?
- What is the total download size of dsh with its dependencies?

## Status

Step 1 and step 2: SKIPPED by user decision. Steps 4 to 9 not started.

Step 3 (test script): `test_toolcalls.py` is complete. The run on 2026-10-27 gave NO result. Ollama (version 0.32.3) hung. The first request ran for 5 minutes and returned HTTP 500. Later requests, even a tiny "hi", timed out. The Ollama runner process 50872 stays stuck. The unload call (keep_alive 0) did not free it. A restart of the Ollama service needs sudo, so the worker did not do it. Expected result from earlier tests: tool called 8/8, first tool correct 6/8. Run the test again after the user restarts Ollama.

Host check, 2026-10-27: `which llama-server dsh` found neither program. Both are not installed.
