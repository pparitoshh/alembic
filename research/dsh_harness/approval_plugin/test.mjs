import test from "node:test";
import assert from "node:assert/strict";
import { Config, apply, inject, name } from "./index.mjs";

const DEFAULT_TOOLS = [
  "mcp__hpc__submit_job",
  "mcp__hpc__cancel_job",
  "mcp__hpc__update_job",
  "mcp__hpc__hold_job",
  "mcp__hpc__release_job",
  "mcp__hpc__requeue_job",
];

function fakeCtx() {
  const registered = [];
  const ctx = {
    on(event, listener) {
      registered.push({ event, listener });
      return () => {};
    },
  };
  return { ctx, registered };
}

function makeListener(config) {
  const { ctx, registered } = fakeCtx();
  apply(ctx, config);
  assert.equal(registered.length, 1, "apply must register exactly one listener");
  assert.equal(registered[0].event, "tools/pre-execute");
  return registered[0].listener;
}

function fakeExec(toolName, args) {
  return { name: toolName, arguments: args, callId: "call-1", agent: {} };
}

async function passThroughResult(listener, toolName, args) {
  let nextCalls = 0;
  const next = () => {
    nextCalls += 1;
    return "NEXT";
  };
  const result = await listener(fakeExec(toolName, args), next);
  return { result, nextCalls };
}

test("exports the cordis plugin shape", () => {
  assert.equal(typeof name, "string");
  assert.ok(name.length > 0);
  assert.equal(typeof apply, "function");
  assert.ok(Array.isArray(inject));
  assert.ok(Config && typeof Config === "object" && Config["~standard"]);
});

test("config validation defaults to the six watched tools and rejects bad values", () => {
  const defaulted = Config["~standard"].validate(undefined);
  assert.deepEqual(defaulted.value.tools, DEFAULT_TOOLS);
  const custom = Config["~standard"].validate({ tools: ["mcp__hpc__cancel_job"] });
  assert.deepEqual(custom.value.tools, ["mcp__hpc__cancel_job"]);
  assert.ok(Config["~standard"].validate({ tools: "cancel_job" }).issues);
  assert.ok(Config["~standard"].validate({ tools: [42] }).issues);
  assert.ok(Config["~standard"].validate("nope").issues);
});

test("asks for every default-listed job-changing tool", async () => {
  const listener = makeListener(undefined);
  const cases = [
    ["mcp__hpc__submit_job", { script: "#!/bin/bash\necho hi", test_only: false }],
    ["mcp__hpc__submit_job", { script: "#!/bin/bash\necho hi" }],
    ["mcp__hpc__cancel_job", { job_id: "42" }],
    ["mcp__hpc__update_job", { job_id: "42", field: "Partition", value: "gpu" }],
    ["mcp__hpc__hold_job", { job_id: "42" }],
    ["mcp__hpc__release_job", { job_id: "42" }],
    ["mcp__hpc__requeue_job", { job_id: "42" }],
  ];
  for (const [toolName, args] of cases) {
    const { result, nextCalls } = await passThroughResult(listener, toolName, args);
    assert.equal(nextCalls, 0, `${toolName} must not call next()`);
    assert.equal(result?.kind, "ask", `${toolName} must ask`);
    assert.match(result.reason, new RegExp(`\\b${toolName}\\b`));
    assert.ok(typeof result.displayReason === "string" && result.displayReason.length > 0);
  }
});

test("passes through tools that are not listed", async () => {
  const listener = makeListener(undefined);
  const { result, nextCalls } = await passThroughResult(listener, "mcp__hpc__run_readonly_command", { command: "squeue" });
  assert.equal(nextCalls, 1);
  assert.equal(result, "NEXT");
});

test("passes through submit_job with test_only true (validation only)", async () => {
  const listener = makeListener(undefined);
  const { result, nextCalls } = await passThroughResult(listener, "mcp__hpc__submit_job", {
    script: "#!/bin/bash\necho hi",
    test_only: true,
  });
  assert.equal(nextCalls, 1);
  assert.equal(result, "NEXT");
});

test("submit_job reason names the tool, the test_only value and at most 300 script characters", async () => {
  const listener = makeListener(undefined);
  const script = "S".repeat(400);
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.equal(result.kind, "ask");
  assert.match(result.reason, /tool mcp__hpc__submit_job/);
  assert.match(result.reason, /test_only=false/);
  assert.ok(result.reason.includes("S".repeat(300)), "reason must carry the first 300 script characters");
  assert.ok(!result.reason.includes("S".repeat(301)), "reason must not carry more than 300 script characters");
  assert.match(result.reason, /\(\+100 chars\)/);
});

test("update_job reason carries job_id, field and value", async () => {
  const listener = makeListener(undefined);
  const { result } = await passThroughResult(listener, "mcp__hpc__update_job", {
    job_id: "42",
    field: "Partition",
    value: "gpu",
  });
  assert.equal(result.kind, "ask");
  for (const part of ["42", "Partition", "gpu"]) {
    assert.ok(result.reason.includes(part), `reason must mention ${part}`);
  }
});

test("submit_job displayReason carries the full script past the 300-character cut", async () => {
  const listener = makeListener(undefined);
  const script = `#!/bin/bash\necho start\n${"X".repeat(400)}\necho done\n`;
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.equal(result.kind, "ask");
  assert.ok(result.displayReason.includes("Full script:"), "displayReason must mark the script block");
  assert.ok(result.displayReason.includes(script), "displayReason must carry the whole script text");
  assert.ok(result.displayReason.includes("X".repeat(400)), "displayReason must keep the >300-char tail");
  assert.ok(!result.displayReason.includes("…(+"), "displayReason must not truncate the script");
  assert.ok(result.reason.includes("…(+"), "reason keeps the truncation marker");
});

test("submit_job displayReason shows the check line and warns when sleeps exceed --time (job 12354 case)", async () => {
  const listener = makeListener(undefined);
  const script = [
    "#!/bin/bash",
    "#SBATCH --time=00:05:00",
    "#SBATCH --mem=1G",
    "echo start",
    "sleep 60",
    "sleep 120",
    "sleep 180",
    "sleep 240",
    "sleep 300",
    "sleep 60",
    "echo done",
  ].join("\n");
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.ok(
    result.displayReason.includes("script 11 lines; --time 5m0s; sleep total 16m0s"),
    `check line missing in: ${result.displayReason}`,
  );
  assert.ok(
    result.displayReason.includes("WARNING: sleep total 16m0s exceeds --time 5m0s"),
    `warning missing in: ${result.displayReason}`,
  );
  assert.ok(result.displayReason.includes("sleep 300"), "the late sleeps past char 300 must be visible");
});

test("submit_job parses every Slurm --time form", async () => {
  const listener = makeListener(undefined);
  const cases = [
    ["90", "1h30m0s"],
    ["1:30", "1m30s"],
    ["2:05", "2m5s"],
    ["1:02:03", "1h2m3s"],
    ["1-02", "1d2h0m0s"],
    ["1-02:03", "1d2h3m0s"],
    ["1-02:03:04", "1d2h3m4s"],
  ];
  for (const [value, formatted] of cases) {
    const script = `#!/bin/bash\n#SBATCH --time=${value}\necho hi\n`;
    const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
    assert.ok(
      result.displayReason.includes(`--time ${formatted}`),
      `--time=${value} must render as ${formatted}, got: ${result.displayReason}`,
    );
    assert.ok(!result.displayReason.includes("WARNING"), `--time=${value} must not warn without sleeps`);
  }
});

test("submit_job accepts --time and -t directive spellings", async () => {
  const listener = makeListener(undefined);
  const directives = [
    "#SBATCH --time=5:00",
    "#SBATCH --time 5:00",
    "#SBATCH -t 5:00",
    "#SBATCH -t=5:00",
    "#SBATCH -t5:00",
  ];
  for (const directive of directives) {
    const script = `#!/bin/bash\n${directive}\necho hi\n`;
    const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
    assert.ok(
      result.displayReason.includes("--time 5m0s"),
      `${directive} must render as --time 5m0s, got: ${result.displayReason}`,
    );
  }
});

test("submit_job sums sleep suffix forms N, Ns, Nm and Nh", async () => {
  const listener = makeListener(undefined);
  const script = ["#!/bin/bash", "#SBATCH --time=00:10:00", "sleep 30", "sleep 45s", "sleep 2m", "sleep 1h"].join("\n");
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.ok(result.displayReason.includes("sleep total 1h3m15s"), `sleep sum wrong in: ${result.displayReason}`);
  assert.ok(
    result.displayReason.includes("WARNING: sleep total 1h3m15s exceeds --time 10m0s"),
    `warning missing in: ${result.displayReason}`,
  );
});

test("submit_job does not warn when sleeps fit the time limit", async () => {
  const listener = makeListener(undefined);
  const script = ["#!/bin/bash", "#SBATCH --time=00:10:00", "sleep 60", "sleep 60"].join("\n");
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.ok(result.displayReason.includes("--time 10m0s"));
  assert.ok(result.displayReason.includes("sleep total 2m0s"));
  assert.ok(!result.displayReason.includes("WARNING"), `unexpected warning in: ${result.displayReason}`);
});

test("submit_job without --time reports no --time and never warns", async () => {
  const listener = makeListener(undefined);
  const script = "#!/bin/bash\nsleep 120\necho hi\n";
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.ok(result.displayReason.includes("no --time"), `check line wrong in: ${result.displayReason}`);
  assert.ok(result.displayReason.includes("sleep total 2m0s"));
  assert.ok(!result.displayReason.includes("WARNING"), `unexpected warning in: ${result.displayReason}`);
});

test("submit_job with an unparsable --time shows the raw value and does not warn", async () => {
  const listener = makeListener(undefined);
  const script = "#!/bin/bash\n#SBATCH --time=unlimited\nsleep 120\necho hi\n";
  const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", { script, test_only: false });
  assert.ok(result.displayReason.includes("--time unlimited"), `raw value missing in: ${result.displayReason}`);
  assert.ok(!result.displayReason.includes("WARNING"), `unexpected warning in: ${result.displayReason}`);
});

test("submit_job with an empty script keeps the old text and no check line", async () => {
  const listener = makeListener(undefined);
  for (const args of [{ script: "" }, {}]) {
    const { result } = await passThroughResult(listener, "mcp__hpc__submit_job", args);
    assert.equal(result.displayReason, "Approve mcp__hpc__submit_job? test_only=false; script: (no script text)");
    assert.ok(!result.displayReason.includes("Full script:"));
  }
});

test("non-submit tools keep the old short displayReason", async () => {
  const listener = makeListener(undefined);
  const cancel = await passThroughResult(listener, "mcp__hpc__cancel_job", { job_id: "42" });
  assert.equal(cancel.result.displayReason, 'Approve mcp__hpc__cancel_job? job_id="42"');
  const update = await passThroughResult(listener, "mcp__hpc__update_job", {
    job_id: "42",
    field: "Partition",
    value: "gpu",
  });
  assert.equal(update.result.displayReason, 'Approve mcp__hpc__update_job? job_id="42", field="Partition", value="gpu"');
});

test("custom tools config asks only for the listed tools", async () => {
  const listener = makeListener({ tools: ["mcp__hpc__cancel_job"] });
  const submit = await passThroughResult(listener, "mcp__hpc__submit_job", { script: "#!/bin/bash" });
  assert.deepEqual(submit, { result: "NEXT", nextCalls: 1 });
  const status = await passThroughResult(listener, "mcp__hpc__job_status", { job_id: "42" });
  assert.deepEqual(status, { result: "NEXT", nextCalls: 1 });
  const cancel = await passThroughResult(listener, "mcp__hpc__cancel_job", { job_id: "42" });
  assert.equal(cancel.nextCalls, 0);
  assert.equal(cancel.result.kind, "ask");
});

test("apply validates its config and rejects a bad tools value", () => {
  assert.throws(() => makeListener({ tools: "cancel_job" }));
});
