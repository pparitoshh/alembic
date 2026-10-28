const SUBMIT_JOB = "mcp__hpc__submit_job";

const DEFAULT_TOOLS = [
  "mcp__hpc__submit_job",
  "mcp__hpc__cancel_job",
  "mcp__hpc__update_job",
  "mcp__hpc__hold_job",
  "mcp__hpc__release_job",
  "mcp__hpc__requeue_job",
];

const SUMMARY_LIMIT = 300;

// `sleep 90`, `sleep 90s`, `sleep 5m`, `sleep 2h`; literal numbers only, no shell variables.
const SLEEP_LINE = /^\s*sleep\s+(\d+(?:\.\d+)?)([smh]?)(?=\s|$)/i;

function truncate(text, limit) {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}…(+${text.length - limit} chars)`;
}

// Slurm --time forms: MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS. Returns seconds or null.
function parseSlurmTime(raw) {
  if (typeof raw !== "string") return null;
  const text = raw.trim().toLowerCase();
  const withDays = text.match(/^(\d+)-(\d+)(?::(\d+))?(?::(\d+))?$/);
  if (withDays) {
    const [, d, h, m = "0", s = "0"] = withDays;
    return Number(d) * 86400 + Number(h) * 3600 + Number(m) * 60 + Number(s);
  }
  const plain = text.match(/^(\d+)(?::(\d+))?(?::(\d+))?$/);
  if (!plain) return null;
  const [, a, b, c] = plain;
  if (c !== undefined) return Number(a) * 3600 + Number(b) * 60 + Number(c);
  if (b !== undefined) return Number(a) * 60 + Number(b);
  return Number(a) * 60;
}

// The value of `#SBATCH --time=V`, `--time V`, `-t=V`, `-t V` or `-tV`, or null.
function findSlurmTimeValue(script) {
  for (const line of script.split("\n")) {
    const match = line.match(/^\s*#\s*SBATCH\s+(?:--time[= ]|-t(?![a-z-])[= ]?)\s*(\S+)/i);
    if (match) return match[1];
  }
  return null;
}

function sleepTotalSeconds(script) {
  let total = 0;
  for (const line of script.split("\n")) {
    const match = line.match(SLEEP_LINE);
    if (!match) continue;
    const unit = match[2].toLowerCase();
    const factor = unit === "h" ? 3600 : unit === "m" ? 60 : 1;
    total += Number(match[1]) * factor;
  }
  return total;
}

function formatDuration(totalSeconds) {
  const total = Math.round(totalSeconds);
  const d = Math.floor(total / 86400);
  const h = Math.floor((total % 86400) / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const parts = [];
  if (d > 0) parts.push(`${d}d`);
  if (d > 0 || h > 0) parts.push(`${h}h`);
  if (d > 0 || h > 0 || m > 0) parts.push(`${m}m`);
  parts.push(`${s}s`);
  return parts.join("");
}

// Check line for the approval panel: line count, --time budget and literal sleep total, plus a
// warning when the sleeps cannot fit into the time limit. The panel headline collapses newlines
// to spaces (no white-space: pre-wrap), so the text stays one block, but nothing is cut.
function submitCheckLine(script) {
  const body = script.endsWith("\n") ? script.slice(0, -1) : script;
  const timeValue = findSlurmTimeValue(script);
  const timeSeconds = timeValue === null ? null : parseSlurmTime(timeValue);
  const sleepTotal = sleepTotalSeconds(script);
  const parts = [`script ${body.split("\n").length} lines`];
  if (timeValue === null) {
    parts.push("no --time");
  } else {
    parts.push(`--time ${timeSeconds === null ? timeValue : formatDuration(timeSeconds)}`);
  }
  parts.push(`sleep total ${formatDuration(sleepTotal)}`);
  const warning =
    timeSeconds !== null && sleepTotal > timeSeconds
      ? `WARNING: sleep total ${formatDuration(sleepTotal)} exceeds --time ${formatDuration(timeSeconds)}`
      : "";
  return { check: parts.join("; "), warning };
}

function summarizeArgs(toolName, args) {
  if (toolName === SUBMIT_JOB) {
    const script = typeof args.script === "string" ? args.script : "";
    const testOnly = args.test_only === true;
    const head = script === "" ? "(no script text)" : truncate(script, SUMMARY_LIMIT);
    return `test_only=${testOnly}; script: ${head}`;
  }
  if (typeof args.job_id !== "undefined") {
    const parts = [`job_id=${JSON.stringify(args.job_id)}`];
    if (typeof args.field !== "undefined") parts.push(`field=${JSON.stringify(args.field)}`);
    if (typeof args.value !== "undefined") parts.push(`value=${JSON.stringify(args.value)}`);
    return parts.join(", ");
  }
  const dump = JSON.stringify(args) ?? "{}";
  return truncate(dump, SUMMARY_LIMIT);
}

function submitDisplayReason(toolName, args) {
  const script = typeof args.script === "string" ? args.script : "";
  const testOnly = args.test_only === true;
  const head = `Approve ${toolName}? test_only=${testOnly}`;
  if (script === "") return `${head}; script: (no script text)`;
  const { check, warning } = submitCheckLine(script);
  const warningLine = warning === "" ? "" : `\n${warning}`;
  return `${head}; ${check}${warningLine}\nFull script:\n${script}`;
}

function describe(toolName, args) {
  const summary = summarizeArgs(toolName, args);
  return {
    reason: `tool ${toolName} requested (${summary})`,
    displayReason:
      toolName === SUBMIT_JOB ? submitDisplayReason(toolName, args) : `Approve ${toolName}? ${summary}`,
  };
}

const Config = {
  ["~standard"]: {
    version: 1,
    validate(value) {
      if (value === undefined || value === null) return { value: { tools: [...DEFAULT_TOOLS] } };
      if (typeof value !== "object" || Array.isArray(value)) {
        return { issues: [{ message: "config must be an object with an optional tools array" }] };
      }
      if (value.tools === undefined) return { value: { ...value, tools: [...DEFAULT_TOOLS] } };
      if (!Array.isArray(value.tools) || !value.tools.every((tool) => typeof tool === "string")) {
        return { issues: [{ message: "tools must be an array of tool name strings", path: ["tools"] }] };
      }
      return { value: { ...value, tools: [...value.tools] } };
    },
  },
};

const inject = [];

function apply(ctx, config) {
  const { value } = Config["~standard"].validate(config);
  const watched = new Set(value.tools);
  ctx.on("tools/pre-execute", async (exec, next) => {
    if (!watched.has(exec.name)) return next();
    if (exec.name === SUBMIT_JOB && exec.arguments?.test_only === true) return next();
    const { reason, displayReason } = describe(exec.name, exec.arguments ?? {});
    return { kind: "ask", reason, displayReason };
  });
}

const name = "hpc-approval";

export { Config, apply, inject, name };
