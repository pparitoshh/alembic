"""Hermetic test environment, applied to every test.

The suite must give the same result on a laptop, in CI and on a Leonardo login node, so a test can't
depend on credentials, the network, the caller's working directory or HOME, or state left behind
by another test. Repo data (configs/, data/) is read through absolute paths from ROOT.
Needs only Python and `bash` (for the shell-syntax check).
"""

import os
import socket

import pytest

from distillkit import tools


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    # no credentials: a test that builds a real Teacher fails instead of making paid API calls
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_TOKEN")) or var in ("OPENAI_BASE_URL", "HF_HOME"):
            monkeypatch.delenv(var)
    # no Hugging Face downloads or cache reads from the real HOME
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # relative paths (run_dir, ./.env) resolve inside the test's own temp dir
    monkeypatch.chdir(tmp_path)

    # no network at all; local subprocesses (bash -n) are unaffected
    def blocked(*args, **kwargs):
        raise RuntimeError("network access in tests is blocked (tests/conftest.py)")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    # module state: the mock sbatch's flag list goes back to the packaged default
    monkeypatch.setattr(tools, "_VALID_FLAGS", None)
