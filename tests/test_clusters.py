"""Cluster profiles: loading and validation of the YAML file, with tmp_path files (no cluster, no network)."""
import pytest

from distillkit.clusters import get_cluster, load_clusters
from distillkit.tools import ToolError

VALID = """
clusters:
  c1:
    ssh: c1.example.org
    login: password
    remote_setup: "module load slurm"
    defaults:
      account: acc1
      partition: part1
  c2:
    ssh: c2
"""


def write(tmp_path, text):
    f = tmp_path / "clusters.yaml"
    f.write_text(text)
    return str(f)


def test_load_valid_file_and_get_cluster(tmp_path):
    path = write(tmp_path, VALID)
    clusters = load_clusters(path)
    assert clusters["c1"] == {"ssh": "c1.example.org", "login": "password", "remote_setup": "module load slurm",
                              "defaults": {"account": "acc1", "partition": "part1"}}
    assert get_cluster("c2", path) == {"ssh": "c2"}


def test_profile_without_ssh_is_rejected(tmp_path):
    with pytest.raises(ToolError, match="missing 'ssh'"):
        load_clusters(write(tmp_path, "clusters:\n  c1:\n    login: password\n"))


def test_unknown_defaults_key_is_rejected(tmp_path):
    text = "clusters:\n  c1:\n    ssh: c1\n    defaults:\n      gpus: 2\n"
    with pytest.raises(ToolError, match="unknown defaults key.*gpus"):
        load_clusters(write(tmp_path, text))


def test_unknown_cluster_lists_the_known_names(tmp_path):
    path = write(tmp_path, VALID)
    with pytest.raises(ToolError, match="c1, c2"):
        get_cluster("nope", path)
