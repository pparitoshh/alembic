# dsh approval plugin (HPC job-changing tool calls)

## Purpose

This plugin asks the user to approve selected MCP tool calls in the dsh web UI. It listens on the
`tools/pre-execute` waterfall of dsh's tools pipeline. For a watched tool call, the listener returns
`{ kind: "ask", reason, displayReason }` without calling `next()`. A listener that does not call
`next()` vetoes the rest of the waterfall chain
(`node_modules/@deepseek-ai/cordis/lib/index.js:305-324` and
`node_modules/@deepseek-ai/dsh-tools/lib/types/index.js:902-907`). dsh then routes the ask through
`serviceAsk` (`dsh-tools/lib/types/index.js:1102`) to the approval service
(`dsh-user-approval` + `dsh-client-ui-approval`), and the user sees Approve/Reject. Reject means
deny. A deployment without an approval service, or a call without an agent, also denies
(`dsh-tools/lib/types/index.js:1104-1125`).

The watched tools are the HPC job-changing actions of the harness MCP server (`serverName: hpc`):
`mcp__hpc__submit_job`, `mcp__hpc__cancel_job`, `mcp__hpc__update_job`, `mcp__hpc__hold_job`,
`mcp__hpc__release_job`, `mcp__hpc__requeue_job`. One call is exempt: `mcp__hpc__submit_job` with
`test_only: true` validates the script only, so the plugin calls `next()` without an ask.

The plugin reads the arguments from `exec.arguments`. This is the real field name; there is no
`exec.args` (`dsh-tools/lib/types/index.js:826` and `:933`). The `reason` string names the tool and
summarizes the arguments: for `submit_job` the `test_only` value and the first 300 characters of the
script (the full script is too long), for the other tools the `job_id` and the optional `field` and
`value`.

## The plugin exports

The module is plain ESM (`index.mjs`, no build step, no npm dependency). It follows the export shape
of a real dsh plugin, `@deepseek-ai/dsh-workspace-changes/lib/index.js:1097`:

```js
export { Config, apply, inject, name };  // name: "hpc-approval", inject: []
```

`Config` implements the Standard Schema interface (`~standard.validate`) that cordis resolves with
`resolveConfig` (`node_modules/@deepseek-ai/cordis/lib/index.js:956-962`). A hand-rolled validator
avoids a dependency on `@deepseek-ai/schemastery`, which the plugin cannot resolve outside the dsh
installation.

Config: `{ tools: string[] }`. The value is the exact list of public tool names. The default is the
six `mcp__hpc__*` job-changing tools above.

## Profile row: loading a local file as "name"

The cordis plugin loader resolves a profile row's `name` in `EntryTree.import`
(`node_modules/@deepseek-ai/cordis-plugin-loader/lib/index.js:214-227`):

```js
import(name, getOuterStack) {
    if (name.startsWith("cordis:")) return this.ctx.loader.builtins[name.slice(7)];
    return composeError(async (info) => {
        info.offset += 3;
        if (this.ctx.loader.internal) return await this.ctx.loader.internal.import(name, this.ctx.baseUrl, {});
        else if (name.startsWith(".")) return await import(__rewriteRelativeImportExtension(
            new URL(name, this.ctx.baseUrl).href
        ));
        else return await import(__rewriteRelativeImportExtension(
            name
        ));
    }, getOuterStack);
}
```

So a row name can be:

1. `cordis:<builtin>` - a loader builtin.
2. A path that starts with `.` - resolved against the loader context's `baseUrl`.
3. Any other string - imported directly by Node's ESM loader (`loader.internal` is Node's own
   cascaded ESM loader, `cordis-plugin-loader/lib/index.js:594`). A direct import accepts an
   absolute POSIX path and a `file://` URL. Both forms were verified with node v24.22.1.

This plugin uses form 3 with an absolute path. Add this row to
`~/.dsh/profiles/web/cordis.patch.yml` and replace `<REPO>` with this repository's path:

```yaml
- id: approval-plugin
  name: <REPO>/research/dsh_harness/approval_plugin/index.mjs
  config:
    tools:
      - mcp__hpc__submit_job
      - mcp__hpc__cancel_job
      - mcp__hpc__update_job
      - mcp__hpc__hold_job
      - mcp__hpc__release_job
      - mcp__hpc__requeue_job
```

A `file://` URL works too: `name: 'file://<REPO>/research/dsh_harness/approval_plugin/index.mjs'`.
Omit the `config` block to get the default tool list. A profile row patch replaces the whole config
of the row with the same id, like the example patch in `research/dsh_harness/cordis.patch.example.yml`.

## Tests

`test.mjs` runs hermetic with plain `node --test test.mjs`. It uses a fake `ctx` that records the
listener, and checks: the ask for each listed tool, the `next()` pass-through for unlisted tools,
the `next()` pass-through for `submit_job` with `test_only: true`, the reason text (tool name,
`test_only` value, first 300 script characters), and a custom `tools` config.
`tests/test_dsh_approval_plugin.py` runs `node --test` on it and skips when node is not on PATH and
not in `~/tools/node/bin`.

## Limits

- The plugin never sees a mode-collapsed call. The collapse denies the call before the pre-execute
  pipeline (`dsh-tools/lib/types/index.js:787-793`).
- The ask degrades to deny without an approval service or without an agent on the call
  (`dsh-tools/lib/types/index.js:1104-1125`). The headless surfaces may have no approval service.
- An earlier `tools/pre-execute` listener that returns its own decision without `next()` wins over
  this plugin: listeners run outermost-first, and this plugin's listener is appended last.
- The tool names depend on the MCP `serverName: hpc` in the profile row. A different server name
  changes the `mcp__hpc__` prefix, and the `tools` config must then list the new names.
- The argument summary trusts the model's arguments: the script is truncated for display, but a full
  approval runs the full script.
- The plugin needs no services, so `inject` is empty. It never runs cluster commands itself.
