"""Runs the dsh approval plugin's `node --test` suite.

The plugin lives in research/dsh_harness/approval_plugin/ and is plain ESM JavaScript, so the
real tests run under node. This wrapper only shells out to `node --test`; it skips when node
is not on PATH and not in ~/tools/node/bin, but only outside CI: GitHub sets CI=true, and
there a missing node must fail the run instead of hiding the suite behind a skip. Node is
resolved at import time, before the hermetic fixture replaces HOME.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO / "research" / "dsh_harness" / "approval_plugin"
TEST_FILE = PLUGIN_DIR / "test.mjs"


def _find_node():
    """The node executable: first on PATH, then ~/tools/node/bin/node."""
    found = shutil.which("node")
    if found:
        return found
    home = os.environ.get("HOME") or str(Path.home())
    candidate = Path(home) / "tools" / "node" / "bin" / "node"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _node_skip(node, ci):
    """The skip reason for the node suite: a missing node only skips the run locally. In CI
    (GitHub sets CI=true; any non-empty value counts) the suite must run and fail instead."""
    if node is not None:
        return None
    return None if ci else "node not found on PATH or in ~/tools/node/bin"


NODE = _find_node()


@pytest.mark.skipif(_node_skip(NODE, os.environ.get("CI")) is not None,
                    reason="node not found on PATH or in ~/tools/node/bin")
def test_approval_plugin_node_suite():
    if NODE is None:  # only reachable in CI: fail with a clear reason instead of a crash on None
        pytest.fail("node not found on PATH or in ~/tools/node/bin (CI requires node for this suite)")
    env = {**os.environ, "PATH": f"{Path(NODE).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    proc = subprocess.run(
        [NODE, "--test", str(TEST_FILE)],
        cwd=str(PLUGIN_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"node --test failed:\n{proc.stdout}\n{proc.stderr}"
    # the clean-run summary: the spec reporter (node >= 22) prints "ℹ fail 0", node 20 prints TAP "# fail 0"
    assert "\nℹ fail 0" in proc.stdout or "\n# fail 0" in proc.stdout, \
        f"node --test did not report a clean run:\n{proc.stdout}"


def test_a_missing_node_skips_locally_but_fails_in_ci():
    assert _node_skip(None, None) == "node not found on PATH or in ~/tools/node/bin"
    assert _node_skip(None, "") == "node not found on PATH or in ~/tools/node/bin"
    assert _node_skip(None, "true") is None  # CI: no skip, the suite runs and fails on the missing node
    assert _node_skip("node", None) is None and _node_skip("node", "true") is None  # node found: never a skip
