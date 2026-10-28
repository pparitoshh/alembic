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
script, for the other tools the `job_id` and the optional `field` and `value`.

For `submit_job` only, the `displayReason` (the text the user reads in the approval panel) differs
from `reason`. It carries the full script text with no cut, preceded by a check line and the script:

```
Approve mcp__hpc__submit_job? test_only=false; script 11 lines; --time 5m0s; sleep total 16m0s
WARNING: sleep total 16m0s exceeds --time 5m0s
Full script:
#!/bin/bash
#SBATCH --time=00:05:00
...
```

The check line gives the script's line count, the `#SBATCH --time` / `-t` value (or `no --time`),
and the sum of all literal `sleep N` values (`N` seconds, also `Ns`, `Nm`, `Nh`). Slurm time forms
`MM`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM`, `D-HH:MM:SS` are parsed. A single value is minutes.
When the sleep total exceeds the time limit, the panel shows
`WARNING: sleep total <S> exceeds --time <T>`. Without a parsable `--time` there is no warning.
This check exists because in job 12354 the user approved a script whose long sleeps
(60+120+180+240+300+60 s in a `--time=00:05:00` job) started after character 300 and were invisible.
The other watched tools keep the plain `Approve <tool>? <summary>` text.

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

This plugin uses form 3 with an absolute path. Add this insert entry to
`~/.dsh/profiles/web/cordis.patch.yml` and replace `<REPO>` with this repository's path:

```yaml
- insert:
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

The insert entry is required: dsh drops a plain row with a NEW id and prints no message. A new row
must be inside an `- insert:` entry. After a restart, check that dsh loaded the row with
`dsh web --dump-config | grep approval-plugin`.

A `file://` URL works too: `name: 'file://<REPO>/research/dsh_harness/approval_plugin/index.mjs'`.
Omit the `config` block to get the default tool list. A profile row patch replaces the whole config
of the row with the same id, like the example patch in `research/dsh_harness/cordis.patch.example.yml`.

## Tests

`test.mjs` runs hermetic with plain `node --test test.mjs`. It uses a fake `ctx` that records the
listener, and checks: the ask for each listed tool, the `next()` pass-through for unlisted tools,
the `next()` pass-through for `submit_job` with `test_only: true`, the reason text (tool name,
`test_only` value, first 300 script characters), the full script and the sleep/`--time` check line in
the `submit_job` `displayReason` (including the warning, all Slurm time forms, `Ns`/`Nm`/`Nh` sleep
suffixes, the within-limit and no-`--time` cases), the unchanged short text of the other tools, and a
custom `tools` config.
`tests/test_dsh_approval_plugin.py` runs `node --test` on it and skips when node is not on PATH and
not in `~/tools/node/bin`.

## Limits

- dsh drops a plain profile row with a NEW id and prints no message. A new row must be inside an
  `- insert:` entry (job 12299 ran with no approval because of this fault).
- The plugin never sees a mode-collapsed call. The collapse denies the call before the pre-execute
  pipeline (`dsh-tools/lib/types/index.js:787-793`).
- The ask degrades to deny without an approval service or without an agent on the call
  (`dsh-tools/lib/types/index.js:1104-1125`). The headless surfaces may have no approval service.
- An earlier `tools/pre-execute` listener that returns its own decision without `next()` wins over
  this plugin: listeners run outermost-first, and this plugin's listener is appended last.
- The tool names depend on the MCP `serverName: hpc` in the profile row. A different server name
  changes the `mcp__hpc__` prefix, and the `tools` config must then list the new names.
- The argument summary trusts the model's arguments: the `reason` string truncates the script at 300
  characters, but a full approval runs the full script. For `submit_job` the panel text
  (`displayReason`) now shows the full script, so nothing is hidden; the check line counts only
  literal `sleep N` lines, so computed sleeps (`sleep "$X"`, loops, `usleep`) stay uncounted.
- The approval panel renders `displayReason` as one text node in a headline div without
  `white-space: pre-wrap` (`dsh-client-ui-approval/lib/client.js:109-111` and the
  `.mna1RW_headline` rule in the same file). Newlines in the text collapse to single spaces on
  screen, so the script shows as one wrapped block, not line-by-line. The full text is still there.
- The plugin needs no services, so `inject` is empty. It never runs cluster commands itself.
