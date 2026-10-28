"""Cluster profiles from one YAML file, so the MCP row only sets HPC_CLUSTER=<name> and the tool adapts by
configuration, not by code per cluster. Standard library, plus PyYAML.

File format (see research/dsh_harness/clusters.example.yaml):
  clusters:
    <name>:
      ssh: <host or ~/.ssh/config alias>   # required
      login: password|one-time-key         # optional, documents how to log in
      remote_setup: <shell snippet>        # optional, e.g. "module load slurm"
      defaults: {account, partition, qos, reservation, gres, time}  # optional sbatch defaults, only for
                                    # options the script does not set in its own #SBATCH lines
"""
import os

import yaml

from .tools import ToolError

CLUSTERS_FILE = "~/.config/hpc-assistant/clusters.yaml"
_DEFAULT_KEYS = {"account", "partition", "qos", "reservation", "gres", "time"}


def load_clusters(path):
    """Read and validate the clusters file: {name: {ssh, login?, remote_setup?, defaults?}}."""
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise ToolError(f"cannot read the cluster file {path}: {e}")
    except yaml.YAMLError as e:
        raise ToolError(f"the cluster file {path} is not valid YAML: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("clusters"), dict):
        raise ToolError(f"the cluster file {path} must have a top-level 'clusters' mapping")
    for name, profile in data["clusters"].items():
        _check_profile(name, profile)
    return data["clusters"]


def get_cluster(name, path=CLUSTERS_FILE):
    """One cluster profile by name. Unknown names get the list of the known ones."""
    clusters = load_clusters(path)
    if name not in clusters:
        known = ", ".join(sorted(clusters)) or "(none)"
        raise ToolError(f"unknown cluster {name!r} in {os.path.expanduser(path)} (known clusters: {known})")
    return clusters[name]


def _check_profile(name, profile):
    if not isinstance(profile, dict):
        raise ToolError(f"cluster {name!r}: the profile must be a mapping")
    if not isinstance(profile.get("ssh"), str) or not profile["ssh"]:
        raise ToolError(f"cluster {name!r}: missing 'ssh' (the host or the ~/.ssh/config alias)")
    for key in ("login", "remote_setup"):
        if profile.get(key) is not None and not isinstance(profile[key], str):
            raise ToolError(f"cluster {name!r}: {key!r} must be a string")
    defaults = profile.get("defaults")
    if defaults is None:
        return
    if not isinstance(defaults, dict):
        raise ToolError(f"cluster {name!r}: 'defaults' must be a mapping")
    if unknown := sorted(set(defaults) - _DEFAULT_KEYS):
        raise ToolError(f"cluster {name!r}: unknown defaults key(s) {unknown} (allowed: {sorted(_DEFAULT_KEYS)})")
