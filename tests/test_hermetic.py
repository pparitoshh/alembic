"""The guarantees in conftest.py and the portability fixes they rely on."""

import os
import socket
from pathlib import Path

import pytest

from distillkit.checks import load_flags
from distillkit.config import load_config
from distillkit.tools import execute, set_valid_flags

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "configs/tools_pilot.yaml"


def test_network_is_blocked():
    with pytest.raises(RuntimeError, match="network access in tests is blocked"):
        socket.create_connection(("example.com", 443), timeout=1)


def test_no_credentials_in_env():
    assert not [v for v in os.environ if v.endswith(("_API_KEY", "_TOKEN"))]


def test_dotenv_is_read_from_cwd_only(tmp_path):
    load_config(CONFIG)
    assert "OPENCODE_API_KEY" not in os.environ  # the repo's .env is not found from config.py's location
    (tmp_path / ".env").write_text("DISTILLKIT_TEST_API_KEY=from-cwd\n")
    load_config(CONFIG)
    assert os.environ.pop("DISTILLKIT_TEST_API_KEY") == "from-cwd"


def test_flag_list_ships_with_the_package():
    flags = load_flags()
    assert {"time", "array", "gpus-per-node", "partition"} <= flags and "gpu-count" not in flags
    assert load_config(CONFIG).verify.flag_list is None  # configs use the packaged list


def test_mock_sbatch_uses_the_run_flag_list(tmp_path):
    script = "#!/bin/bash\n#SBATCH --time=01:00:00\nsrun hostname"
    assert execute("submit_job", {"script": script})["submitted"]
    set_valid_flags({"partition"})  # a cluster whose sbatch has no --time
    assert execute("submit_job", {"script": script})["error"] == "sbatch: unrecognized option '--time'"


def test_mock_sbatch_flags_reset_between_tests():
    # runs after the test above in file order: conftest restores the packaged list
    assert execute("submit_job", {"script": "#!/bin/bash\n#SBATCH --time=01:00:00\nsrun hostname"})["submitted"]
