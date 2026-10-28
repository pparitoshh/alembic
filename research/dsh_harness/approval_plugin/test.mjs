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
