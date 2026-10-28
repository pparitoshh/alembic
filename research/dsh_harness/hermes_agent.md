# Hermes Agent as a local harness for the HPC assistant

Research date: 2026-10-27. Method: WebFetch of official docs and the official GitHub page.
Each fact has a tag: (official) or (third-party). "not found" means the sources did not show the fact.

## 1. What it is
- Hermes Agent is a self-improving AI agent from Nous Research. It has skills, persistent memory and past-chat search. (official, GitHub README)
- Licence: MIT. (official)
- Version: not found. The GitHub page showed no release or version number.
- Maturity: about 257.7k stars, 56.5k forks, about 59,600 commits, more than 5k open issues. (official, GitHub page)

## 2. Install
- Linux, macOS, WSL2: `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash` (official)
- Windows: `iex (irm https://hermes-agent.nousresearch.com/install.ps1)` (official)
- Pinned version: not found. The documented command has no version option.
- Download size: not found.
- The install script downloads code from the network. Review it before you run it.

## 3. Custom OpenAI-compatible endpoint (official, docs "providers" page)
Config file: `~/.hermes/config.yaml`. Secrets go in `~/.hermes/.env`.
```yaml
model:
  default: hpc-assistant-vkkm-live:latest
  provider: custom
  base_url: http://127.0.0.1:11434/v1
  api_key:            # empty is allowed for local servers
  context_length: 64000
```
- You can also run `hermes model` and choose "Custom endpoint".
- Important: Hermes requires a context window of at least 64,000 tokens. It rejects smaller windows at start. (official)
- The Ollama default context is small. Set `OLLAMA_CONTEXT_LENGTH=64000` when you start `ollama serve`, or set `num_ctx` in a Modelfile. (official)
- Risk (our inference, not from docs): a 64k context for Qwen3-4B needs more memory for the KV cache. Check `ollama ps`.

## 4. Custom tools
- MCP server (official): add to `mcp_servers` in `~/.hermes/config.yaml`.
```yaml
mcp_servers:
  slurm:
    command: "python"
    args: ["/path/to/slurm_mcp_server.py"]
```
  The docs show the `command` and `args` form with the `npx` filesystem server. The Slurm server is our example. Optional `tools: include/exclude` lists filter the tools. Hermes names each tool `mcp_<server>_<tool>`. Reload with `/reload-mcp`.
- Python tool (official, developer guide): a file in `tools/*.py` calls `registry.register(name=..., toolset=..., schema=..., handler=..., check_fn=..., requires_env=[...])`. The registry finds it at import. This changes the Hermes source tree. No full example was available to me.
- Plugin: the docs say plugins are the preferred route for custom tools. Details of plugin tool registration: not found. Install command: `hermes plugins install` (official).
- Our 18 OpenAI-format schemas: the `schema` argument of `registry.register` probably takes a tool schema (our inference). The simplest path is a small MCP server that exposes the 18 tools. Loading OpenAI-format schemas directly from a JSON file: not found.

## 5. Tool-call parsing
- Hermes uses structured `tool_calls` from the provider. All API modes (chat_completions, codex_responses, anthropic_messages) convert to OpenAI-style `tool_calls`. (official, agent-loop page)
- Client-side parsing of the `<tool_call>{...}</tool_call>` text: not found in the docs.
- For vLLM the docs say that without `--enable-auto-tool-choice` and `--tool-call-parser hermes`, tool calls return as plain text. (official) So the server must parse.
- For Ollama: tool calling is on by default. Check `ollama show <model>`. (official) Ollama converts the Qwen3 `<tool_call>` text into `tool_calls` only if the model template supports tools. Verify this for our model (our inference).

## 6. UI
- Terminal UI with multiline edit, slash commands and streaming tool output. (official)
- Messaging gateway: Telegram, Discord, Slack, WhatsApp, Signal, Email. (official)
- A separate Hermes Desktop app exists. (official, linked from README)
- A built-in local web UI: not found.

## 7. Telemetry and logging
- Logs: `~/.hermes/logs/` (`errors.log`, `gateway.log`). Secrets are auto-redacted. (official)
- Sessions: SQLite `state.db` with full-text search, and `~/.hermes/sessions/`. Retention `sessions.retention_days`, default 90. (official)
- Update check: calls the GitHub REST API at most once each 24 hours. (official)
- Telemetry statement: not found in the official pages that I read (README, security, MCP, providers, and the first part of the config page). About 180k characters of the config page were not read.
- A third-party plugin (Latitude) can send traces off the machine. It is optional. (third-party)
- Nous Portal sends prompts to Nous servers. (third-party). With a local Ollama endpoint, model traffic stays local (our inference).
- How to turn off telemetry: not found.

## 8. Approval policy (official, security page)
- Config key `approvals.mode` in `~/.hermes/config.yaml`: `smart` (default), `manual`, `off`.
- Approval applies to shell commands that match a dangerous-pattern list (for example `rm -r`, `systemctl stop`, `pkill -9`). `approvals.deny` takes glob patterns that always block.
- Slurm `scancel` or `scontrol hold` through an MCP tool: no documented approval rule. Not found. An MCP tool is not a shell command, so the pattern list probably does not apply (our inference). Add the confirmation in our own tool code.
- `smart` mode uses an auxiliary LLM to judge risk. With a 4B model, prefer `manual`.

## 9. Comparison with DeepSeek Harness (dsh)
The dsh column uses the "dsh facts" table in `research/dsh_harness/plan.md`.

| Item | Hermes Agent | dsh |
|---|---|---|
| Install size | not found | 75,972 bytes unpacked for the package alone, 86 direct dependencies; total not found |
| UI | Terminal, messaging gateway, desktop app | Web UI at http://127.0.0.1:3080 |
| Tool registration | MCP config, Python registry, plugins | Plugin (tool form plus `dsh-plugin.naming.json`); MCP server by profile patch |
| Telemetry | none documented; local logs and SQLite | Built on the session log; data destination not found |
| Maturity | MIT, 257k stars, version not found | MIT, Developer Preview, npm 0.2.1-rc.2 |

## 10. Recommendation
1. Hermes is mature and supports a custom Ollama endpoint with simple YAML.
2. It is large (more than 40 built-in tools, memory, gateway) and may fit a 4B model poorly.
3. The 64k minimum context may stress Qwen3-4B on our hardware.
4. Use a Slurm MCP server for the 18 tools and set `approvals.mode: manual`.
5. Run a small test first. Decide after you read the dsh facts.
