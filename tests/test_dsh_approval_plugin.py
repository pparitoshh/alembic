"""Runs the dsh approval plugin's `node --test` suite.

The plugin lives in research/dsh_harness/approval_plugin/ and is plain ESM JavaScript, so the
real tests run under node. This wrapper only shells out to `node --test`; it skips when node
is not on PATH and not in ~/tools/node/bin. Node is resolved at import time, before the
hermetic fixture replaces HOME.
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


NODE = _find_node()

pytestmark = pytest.mark.skipif(NODE is None, reason="node not found on PATH or in ~/tools/node/bin")


def test_approval_plugin_node_suite():
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
    assert "\nℹ fail 0" in proc.stdout, f"node --test did not report a clean run:\n{proc.stdout}"
