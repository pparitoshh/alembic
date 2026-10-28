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

function truncate(text, limit) {
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}…(+${text.length - limit} chars)`;
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

function describe(toolName, args) {
  const summary = summarizeArgs(toolName, args);
  return {
    reason: `tool ${toolName} requested (${summary})`,
    displayReason: `Approve ${toolName}? ${summary}`,
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
