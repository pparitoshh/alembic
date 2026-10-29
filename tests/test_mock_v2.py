"""Mock cluster v2 (issues #114, #112): Leonardo's partitions with QOS lists, and the submit-time limits of a real sbatch.
v1 stays the default and unchanged for archived runs."""
import hashlib
import json

import pytest

from distillkit.tools import MOCK_VERSION, MOCK_VERSION_V2, PARTITIONS_V2, execute, mock_session

HEAD = "#!/bin/bash\n"


def submit(lines, test_only=True):
    with mock_session(mock_version=MOCK_VERSION_V2):
        return execute("submit_job", {"script": HEAD + "\n".join(lines) + "\nsrun hostname", "test_only": test_only})


def test_v1_behaviour_is_unchanged():
    """The same battery of calls as before v2 existed gives byte-identical results (hash recorded on dev 32d511e)."""
    calls = [("partition_info", {}), ("partition_info", {"partition": "boost_qos_dbg"}), ("partition_info", {"partition": "x"}),
             ("gpu_availability", {}), ("gpu_availability", {"partition": "boost_qos_lprod"}), ("list_queue", {}),
             ("job_status", {"job_id": "4123456"}), ("job_accounting", {"job_id": "4123457"}),
             ("read_job_log", {"job_id": "4123458"}), ("cancel_job", {"job_id": "4123459"}),
             ("submit_job", {"script": "#!/bin/bash\n#SBATCH -p boost_qos_dbg\n#SBATCH --cpus-per-task=64\n#SBATCH -t 30:00:00\nhostname"}),
             ("submit_job", {"script": "#!/bin/bash\n#SBATCH --partition=boost_usr_prod\n#SBATCH --nodes=9\n#SBATCH --mem=900G\nhostname",
                             "test_only": True}),
             ("submit_job", {"script": "#!/bin/bash\n#SBATCH -p nope\nhostname"})]
    with mock_session():
        out = [execute(n, a) for n, a in calls] + [execute("job_status", {"job_id": str(4100000 + i)}) for i in range(60)]
    digest = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    assert digest == "de4cc45391e63cc602cca88568ae8ae98ff1346ce115b5aca05514ade8f5820d"


def test_unknown_mock_version_is_refused():
    with pytest.raises(ValueError):
        with mock_session(mock_version="v3"):
            pass


def test_partition_info_lists_real_partitions_with_qos():
    with mock_session(mock_version=MOCK_VERSION_V2):
        parts = execute("partition_info", {})["partitions"]
        boost = execute("partition_info", {"partition": "boost_usr_prod"})
        missing = execute("partition_info", {"partition": "boost_qos_dbg"})
    assert [p["partition"] for p in parts] == ["boost_usr_prod", "dcgp_usr_prod", "lrd_all_serial"]
    qos = {q["qos"]: q for q in boost["qos"]}
    assert qos["boost_qos_dbg"]["max_time"] == "00:30:00" and qos["boost_qos_dbg"]["max_nodes"] == 8
    assert qos["boost_qos_lprod"]["max_time"] == "4-00:00:00" and qos["boost_qos_lprod"]["max_nodes"] == 8
    assert qos["normal"]["max_nodes"] == 64 and boost["cpus_per_node"] == 32 and boost["gpus_per_node"] == 4
    assert "error" in missing


def test_v2_job_fixtures_use_the_debug_qos_on_boost_usr_prod():
    with mock_session(mock_version=MOCK_VERSION_V2):
        jobs = [execute("job_status", {"job_id": str(4100000 + i)}) for i in range(60)]
    assert all(j["partition"] in PARTITIONS_V2 for j in jobs)
    assert any(j.get("qos") == "boost_qos_dbg" and j["partition"] == "boost_usr_prod" for j in jobs)


@pytest.mark.parametrize("lines, error", [
    # QOS names are not partitions (#114)
    (["#SBATCH -p boost_qos_dbg"], "invalid partition specified: boost_qos_dbg"),
    (["#SBATCH --partition=boost_qos_lprod"], "invalid partition specified: boost_qos_lprod"),
    (["#SBATCH -p booster"], "invalid partition specified: booster"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_fast"], "Invalid qos specification"),
    # node size (#112: 64 CPUs on a 32-core boost node, all of its memory, 6 GPUs)
    (["#SBATCH -p boost_usr_prod", "#SBATCH --cpus-per-task=64"], "Requested node configuration is not available"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --ntasks-per-node=8", "#SBATCH --cpus-per-task=8"], "Requested node configuration is not available"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --ntasks=64"], "Requested node configuration is not available"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --mem=502G"], "Requested node configuration is not available"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --mem-per-cpu=20G", "#SBATCH --cpus-per-task=32"], "Requested node configuration is not available"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --gres=gpu:6"], "Requested node configuration is not available"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --gpus-per-node=8"], "Requested node configuration is not available"),
    (["#SBATCH -p dcgp_usr_prod", "#SBATCH --cpus-per-task=128"], "Requested node configuration is not available"),
    (["#SBATCH -p dcgp_usr_prod", "#SBATCH --gres=gpu:1"], "Invalid generic resource (gres) specification"),
    # wall time per QOS
    (["#SBATCH -p boost_usr_prod", "#SBATCH -t 30:00:00"], "QOSMaxWallDurationPerJobLimit"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_dbg", "#SBATCH --time=01:00:00"], "QOSMaxWallDurationPerJobLimit"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_lprod", "#SBATCH --time=5-00:00:00"], "QOSMaxWallDurationPerJobLimit"),
    (["#SBATCH -t 5:00:00"], "QOSMaxWallDurationPerJobLimit"),  # default partition lrd_all_serial: 4 h
    # nodes per QOS
    (["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_dbg", "#SBATCH --nodes=9"], "QOSMaxNodePerJobLimit"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_lprod", "#SBATCH -N 9"], "MaxNodePerAccount"),
    (["#SBATCH -p boost_usr_prod", "#SBATCH --nodes=65"], "QOSMaxNodePerJobLimit"),
    (["#SBATCH --nodes=2"], "QOSMaxNodePerUserLimit"),  # no partition: lrd_all_serial, as on Leonardo
])
def test_v2_rejects_with_the_real_error(lines, error):
    res = submit(lines)
    assert "error" in res and error in res["error"], res


@pytest.mark.parametrize("lines", [
    ["#SBATCH -p boost_usr_prod", "#SBATCH --nodes=2", "#SBATCH --ntasks-per-node=4", "#SBATCH --cpus-per-task=8",
     "#SBATCH --gres=gpu:4", "#SBATCH -t 24:00:00"],  # #111's layout: one rank per GPU on 2 nodes
    ["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_dbg", "#SBATCH -N 8", "#SBATCH -t 00:30:00"],
    ["#SBATCH -p boost_usr_prod", "#SBATCH --qos=boost_qos_lprod", "#SBATCH --nodes=8", "#SBATCH --time=4-00:00:00"],
    ["#SBATCH -p boost_usr_prod", "#SBATCH --mem=120G", "#SBATCH --cpus-per-task=8", "#SBATCH --gpus-per-node=1"],
    ["#SBATCH -p dcgp_usr_prod", "#SBATCH --nodes=1", "#SBATCH --ntasks=64"],  # #112's case on the CPU partition
    ["#SBATCH -p dcgp_usr_prod", "#SBATCH --nodes=2", "#SBATCH --ntasks=200"],
    ["#SBATCH --cpus-per-task=4", "#SBATCH -t 04:00:00"],  # serial default partition
    ["#SBATCH -p boost_usr_prod", "#SBATCH --nodes=2-4", "#SBATCH -t 90"],  # node range, minutes
])
def test_v2_accepts_valid_scripts(lines):
    res = submit(lines)
    assert res.get("valid") is True and "qos_name" in res["validation_scope"], res


def test_v2_submit_records_partition_and_qos():
    with mock_session(mock_version=MOCK_VERSION_V2):
        res = execute("submit_job", {"script": HEAD + "#SBATCH -p boost_usr_prod\n#SBATCH -q boost_qos_dbg\n#SBATCH -t 10\nhostname"})
        job = execute("job_status", {"job_id": res["job_id"]})
    assert job["partition"] == "boost_usr_prod" and job["qos"] == "boost_qos_dbg" and job["state"] == "PENDING"


def test_v1_still_accepts_qos_partitions():
    with mock_session(mock_version=MOCK_VERSION):  # archived traces keep their results
        assert execute("submit_job", {"script": HEAD + "#SBATCH -p boost_qos_dbg\nhostname", "test_only": True})["valid"]
