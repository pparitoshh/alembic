"""Live Slurm backend: command construction and output parsing against recorded Slurm output (no cluster needed)."""
import json
import shlex
from io import StringIO

import pytest

from distillkit import assistant, slurm_live
from distillkit.slurm_live import SlurmLive
from distillkit.tools import SCHEMAS, ToolError


class FakeSlurm:
    """Maps a command (argv[0] plus a distinguishing word) to recorded output; records every call."""

    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def __call__(self, argv, stdin=None):
        self.calls.append((argv, stdin))
        joined = " ".join(argv)
        for key, resp in self.responses.items():
            if all(k in joined for k in key.split("&")):
                return resp
        return 1, "", f"unexpected command: {joined}"


def live(responses, confirm=lambda n, a: True):
    fake = FakeSlurm(responses)
    return SlurmLive(runner=fake, confirm=confirm, workdir="/w/jobs"), fake


QUEUE_ARGV = ["squeue", "--me", "-h", "-o", "%i|%j|%T|%P|%M|%l|%r", "-t", "RUNNING"]

RESERVATIONS = """
ReservationName=s_tra_ncc2 StartTime=2026-10-28T08:00:00 EndTime=2026-10-28T22:00:00 Duration=14:00:00
   Nodes=lrdn[0066,0083] NodeCnt=32 CoreCnt=1024 Features=(null) PartitionName=boost_usr_prod Flags=
   Users=(null) Groups=(null) Accounts=EUHPC_D30_031 Licenses=(null) State=ACTIVE BurstBuffer=(null)

ReservationName=s_tra_ncc3 StartTime=2026-10-29T08:00:00 EndTime=2026-10-29T15:00:00 Duration=07:00:00
   Nodes=lrdn[0066,0083] NodeCnt=32 CoreCnt=1024 Features=(null) PartitionName=boost_usr_prod Flags=
   Users=(null) Groups=(null) Accounts=EUHPC_D30_031 Licenses=(null) State=INACTIVE BurstBuffer=(null)
"""


def test_live_schemas_are_the_trained_ones_plus_live_tools():
    assert slurm_live.LIVE_SCHEMAS[:len(SCHEMAS)] == SCHEMAS
    assert [t["function"]["name"] for t in slurm_live.openai_tools()][len(SCHEMAS):] == \
        ["job_script", "job_usage", "job_priority", "update_job", "hold_job", "release_job", "requeue_job",
         "run_readonly_command", "reservation_info"]


def test_job_status_running_pending_and_finished():
    s, _ = live({"squeue&-j&42": (0, "42|train|RUNNING|None|boost_usr_prod|2|1:02:03|4:00:00|lrdn[0001-0002]|N/A\n", ""),
                 "squeue&-j&43": (0, "43|eval|PENDING|Priority|boost_usr_prod|1|0:00|1:00:00||2026-10-26T10:00:00\n", "")})
    assert s.execute("job_status", {"job_id": "42"}) == {
        "job_id": "42", "name": "train", "state": "RUNNING", "partition": "boost_usr_prod", "nodes": 2,
        "elapsed": "1:02:03", "time_limit": "4:00:00", "node_list": "lrdn[0001-0002]"}
    p = s.execute("job_status", {"job_id": "43"})
    assert p["reason"] == "Priority" and p["est_start"] == "2026-10-26T10:00:00" and "node_list" not in p
    s, _ = live({"squeue": (0, "", ""), "sacct": (0, "44|run|TIMEOUT|0:15|boost_usr_prod|04:00:12|04:00:00|lrdn0007||"
                                                     "14:00:00|8|billing=8,cpu=8,gres/gpu=1,mem=60G,node=1\n", "")})
    f = s.execute("job_status", {"job_id": "44"})
    assert f["state"] == "TIMEOUT" and f["exit_code"] == "0:15" and f["time_limit"] == "04:00:00"


def test_job_accounting_efficiency_memory_and_gpus():
    s, _ = live({"sacct": (0, "7|job|COMPLETED|0:0|p|01:00:00|02:00:00|n1||02:00:00|4|cpu=4,gres/gpu:a100=2,node=1\n"
                              "7.batch|batch|COMPLETED|0:0||01:00:00||n1|3145728K|02:00:00|4|\n", "")})
    a = s.execute("job_accounting", {"job_id": "7"})
    assert a["cpu_efficiency"] == "50%" and a["max_rss"] == "3.0G" and a["gpus"] == 2


def test_submit_writes_the_script_then_sbatch_and_test_only():
    s, fake = live({"mkdir": (0, "", ""), "sbatch&--test-only": (0, "", "sbatch: Job 9 to start at 2026-10-26T10:00:00"),
                    "sbatch": (0, "123;cluster\n", "")})
    r = s.execute("submit_job", {"script": "#!/bin/bash\n#SBATCH -t 5\necho hi\n"})
    assert r["submitted"] and r["job_id"] == "123"
    assert fake.calls[0][1] == "#!/bin/bash\n#SBATCH -t 5\necho hi\n"  # the script goes over stdin, never into argv
    t = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n", "test_only": True})
    assert t["valid"] and "start at" in t["message"]
    assert "error" in s.execute("submit_job", {"script": "echo no shebang"})


def test_submit_job_rejects_a_path_or_url_as_the_script():
    for script in ["/home/me/job.sh", "~/job.sbatch", "./run.py", "job.slurm",
                   "http://example.com/job.sh", "https://example.com/job.sh"]:
        s, fake = live({})
        r = s.execute("submit_job", {"script": script})
        assert r["error"] == ("script must be the full text of the batch script, starting with #! and #SBATCH lines. "
                              "A path or URL does not work: the server cannot read your files. "
                              "Ask the user for the script text."), script
        assert fake.calls == []  # nothing written or run
    # a one-line script that is none of those shapes keeps the old sbatch error
    s, fake = live({})
    r = s.execute("submit_job", {"script": "echo no shebang"})
    assert "This does not look like a batch script" in r["error"]
    assert fake.calls == []


def test_actions_need_confirmation_and_cancel_only_own_jobs():
    s, fake = live({"sbatch": (0, "1\n", "")}, confirm=lambda n, a: False)
    assert s.execute("submit_job", {"script": "#!/bin/bash\n"})["error"] == "the user declined this action"
    assert not any("sbatch" in " ".join(c[0]) for c in fake.calls)
    s, fake = live({"squeue&--me": (0, "", "")})
    assert s.execute("cancel_job", {"job_id": "5"})["cancelled"] is False
    assert not any(c[0][0] == "scancel" for c in fake.calls)
    s, _ = live({"squeue&--me&%i": (0, "5\n", ""), "squeue&-j&5": (0, "5|x|PENDING|Priority|p|1|0:00|1:00||N/A\n", ""),
                 "scancel": (0, "", "")})
    assert s.execute("cancel_job", {"job_id": "5"}) == {"cancelled": True, "message": "Job 5 cancelled (was PENDING)"}


def test_submit_job_resolves_the_auto_reservation_to_the_one_active_match():
    responses = {"mkdir": (0, "", ""), "sbatch": (0, "123;cluster\n", ""),
                 "scontrol&show&reservation": (0, RESERVATIONS, "")}
    fake = FakeSlurm(responses)
    s = SlurmLive(runner=fake, job_defaults={"account": "acc1", "partition": "part1", "reservation": "auto:s_tra_ncc"},
                  confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\n#SBATCH -t 5\n"})
    assert r["submitted"] and "note" not in r
    assert r["defaults_applied"] == ["--account=acc1", "--partition=part1", "--reservation=s_tra_ncc2"]
    assert "--reservation=s_tra_ncc2" in " ".join(fake.calls[2][0])  # scontrol, mkdir, then sbatch


def test_submit_job_auto_reservation_with_no_active_match_is_an_error():
    fake = FakeSlurm({"scontrol&show&reservation": (0, RESERVATIONS.replace("State=ACTIVE", "State=INACTIVE"), "")})
    s = SlurmLive(runner=fake, job_defaults={"reservation": "auto:s_tra_ncc"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n"})
    assert r == {"error": "no ACTIVE reservation that starts with s_tra_ncc; the job was not submitted"}
    assert fake.calls == [(["scontrol", "show", "reservation"], None)]  # no sbatch and no script write


def test_submit_job_auto_reservation_with_two_actives_lists_them():
    fake = FakeSlurm({"scontrol&show&reservation": (0, RESERVATIONS.replace("State=INACTIVE", "State=ACTIVE"), "")})
    s = SlurmLive(runner=fake, job_defaults={"reservation": "auto:s_tra_ncc"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n"})
    assert "s_tra_ncc2, s_tra_ncc3" in r["error"]
    assert fake.calls == [(["scontrol", "show", "reservation"], None)]


def test_submit_job_auto_reservation_with_no_reservations_at_all_is_an_error():
    fake = FakeSlurm({"scontrol&show&reservation": (0, "No reservations in the system\n", "")})
    s = SlurmLive(runner=fake, job_defaults={"reservation": "auto:s_tra_ncc"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n"})
    assert r == {"error": "no ACTIVE reservation that starts with s_tra_ncc; the job was not submitted"}
    assert fake.calls == [(["scontrol", "show", "reservation"], None)]


def test_submit_job_failing_scontrol_is_an_error():
    fake = FakeSlurm({"scontrol&show&reservation": (1, "", "scontrol: error: something broke")})
    s = SlurmLive(runner=fake, job_defaults={"reservation": "auto:s_tra_ncc"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n"})
    assert "scontrol show reservation failed" in r["error"] and "something broke" in r["error"]
    assert fake.calls == [(["scontrol", "show", "reservation"], None)]


def test_reservation_info_lists_all_reservations_with_their_jobs():
    s, fake = live({"scontrol&show&reservation": (0, RESERVATIONS, ""),
                    "squeue&-R&s_tra_ncc2": (0, "101|train|alice|RUNNING|01:23:45\n102|eval|bob|PENDING|0:00\n", ""),
                    "squeue&-R&s_tra_ncc3": (0, "", "")})
    r = s.execute("reservation_info", {})
    assert r == {"reservations": [
        {"name": "s_tra_ncc2", "state": "ACTIVE", "start": "2026-10-28T08:00:00", "end": "2026-10-28T22:00:00",
         "partition": "boost_usr_prod", "node_count": 32, "accounts": "EUHPC_D30_031",
         "jobs": [{"job_id": "101", "name": "train", "user": "alice", "state": "RUNNING", "time": "01:23:45"},
                  {"job_id": "102", "name": "eval", "user": "bob", "state": "PENDING", "time": "0:00"}]},
        {"name": "s_tra_ncc3", "state": "INACTIVE", "start": "2026-10-29T08:00:00", "end": "2026-10-29T15:00:00",
         "partition": "boost_usr_prod", "node_count": 32, "accounts": "EUHPC_D30_031", "jobs": []}]}
    assert fake.calls[0][0] == ["scontrol", "show", "reservation"]


def test_reservation_info_by_name_queries_squeue_with_R_and_lists_its_jobs():
    s, fake = live({"scontrol&show&reservation": (0, RESERVATIONS, ""),
                    "squeue&-R&s_tra_ncc2": (0, "101|train|alice|RUNNING|01:23:45\n102|eval|bob|PENDING|0:00\n", "")})
    r = s.execute("reservation_info", {"name": "s_tra_ncc2"})
    assert len(r["reservations"]) == 1 and r["reservations"][0]["state"] == "ACTIVE"
    assert r["reservations"][0]["jobs"] == [
        {"job_id": "101", "name": "train", "user": "alice", "state": "RUNNING", "time": "01:23:45"},
        {"job_id": "102", "name": "eval", "user": "bob", "state": "PENDING", "time": "0:00"}]
    assert [c[0] for c in fake.calls if c[0][0] == "squeue"] == \
        [["squeue", "-R", "s_tra_ncc2", "-h", "-o", "%i|%j|%u|%T|%M"]]


def test_reservation_info_with_no_reservations_returns_an_empty_list():
    s, _ = live({"scontrol&show&reservation": (0, "No reservations in the system\n", "")})
    assert s.execute("reservation_info", {}) == {"reservations": []}


def test_reservation_info_unknown_name_is_an_error_without_a_squeue_call():
    s, fake = live({"scontrol&show&reservation": (0, RESERVATIONS, "")})
    assert s.execute("reservation_info", {"name": "s_tra_ncc9"}) == {"error": "Reservation s_tra_ncc9 not found"}
    assert [c[0][0] for c in fake.calls] == ["scontrol"]


def test_submit_job_test_only_resolves_the_auto_reservation_the_same_way():
    responses = {"mkdir": (0, "", ""), "sbatch&--test-only": (0, "", "sbatch: Job 9 to start at 2026-10-28T09:00:00"),
                 "scontrol&show&reservation": (0, RESERVATIONS, "")}
    fake = FakeSlurm(responses)
    s = SlurmLive(runner=fake, job_defaults={"reservation": "auto:s_tra_ncc"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n", "test_only": True})
    assert r["valid"] and r["defaults_applied"] == ["--reservation=s_tra_ncc2"]
    assert "--reservation=s_tra_ncc2" in " ".join(fake.calls[2][0])


def test_submit_job_the_scripts_own_reservation_skips_the_lookup():
    fake = FakeSlurm({"mkdir": (0, "", ""), "sbatch": (0, "123;cluster\n", "")})
    s = SlurmLive(runner=fake, job_defaults={"reservation": "auto:s_tra_ncc"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\n#SBATCH --reservation=x\necho hi\n"})
    assert r["submitted"] and "defaults_applied" not in r
    assert not any("scontrol" in " ".join(c[0]) for c in fake.calls)
    assert "--reservation" not in " ".join(fake.calls[1][0])


def test_submit_job_the_scripts_own_sbatch_lines_block_the_defaults():
    fake = FakeSlurm({"mkdir": (0, "", ""), "sbatch": (0, "123;cluster\n", "")})
    s = SlurmLive(runner=fake, job_defaults={"account": "other", "partition": "part1"}, confirm=lambda n, a: True)
    script = "#!/bin/bash\n#SBATCH --account=mine\n#SBATCH -p dbg\necho hi\n"
    r = s.execute("submit_job", {"script": script})
    sbatch = " ".join(fake.calls[1][0])
    assert r["submitted"] and "defaults_applied" not in r
    assert "--account" not in sbatch and "--partition" not in sbatch


def test_submit_job_adds_the_defaults_the_script_does_not_set():
    fake = FakeSlurm({"mkdir": (0, "", ""), "sbatch": (0, "123;cluster\n", "")})
    s = SlurmLive(runner=fake, job_defaults={"account": "acc1", "partition": "part1", "time": "30"},
                  confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\n#SBATCH --time=10\necho hi\n"})
    sbatch = " ".join(fake.calls[1][0])
    assert r["submitted"] and r["defaults_applied"] == ["--account=acc1", "--partition=part1"]
    assert "--account=acc1" in sbatch and "--partition=part1" in sbatch and "--time" not in sbatch


def test_submit_job_ignores_sbatch_lines_after_the_first_command():
    fake = FakeSlurm({"mkdir": (0, "", ""), "sbatch": (0, "123;cluster\n", "")})
    s = SlurmLive(runner=fake, job_defaults={"partition": "part1"}, confirm=lambda n, a: True)
    r = s.execute("submit_job", {"script": "#!/bin/bash\necho hi\n#SBATCH -p dbg\n"})
    assert r["defaults_applied"] == ["--partition=part1"]
    assert "--partition=part1" in " ".join(fake.calls[1][0])


def test_submit_job_with_defaults_still_needs_confirmation():
    fake = FakeSlurm({"mkdir": (0, "", "")})
    s = SlurmLive(runner=fake, job_defaults={"account": "acc1"})  # the default confirm declines
    r = s.execute("submit_job", {"script": "#!/bin/bash\n"})
    assert r["error"] == "the user declined this action"
    assert not any("sbatch" in " ".join(c[0]) for c in fake.calls)


def test_read_job_log_uses_the_jobs_own_paths():
    s, fake = live({"scontrol&show&job": (0, "JobId=8 StdErr=/w/slurm-8.out StdIn=/dev/null StdOut=/w/slurm-8.out\n", ""),
                    "tail": (0, "line1\nline2\n", "")})
    r = s.execute("read_job_log", {"job_id": "8", "stream": "stderr", "tail_lines": 2})
    assert r["file"] == "/w/slurm-8.out" and r["lines"] == ["line1", "line2"] and "same file" in r["note"]
    assert ["tail", "-n", "2", "/w/slurm-8.out"] in [c[0] for c in fake.calls]
    with pytest.raises(ToolError):
        s.execute("read_job_log", {"job_id": "8", "tail_lines": 500})


def test_read_job_log_of_a_finished_job_expands_the_sacct_pattern():
    # As on Leonardo (Slurm 25.05): --output=slurm/logs/%x-%j.out and no --error, so sacct's StdErr is empty.
    sacct = "/w/repo/slurm/logs/%x-%j.out||/w/repo|dk-export|a08trc06|60340010\n"
    s, fake = live({"scontrol&show&job": (1, "", "slurm_load_jobs error: Invalid job id specified"),
                    "sacct&StdOut": (0, sacct, ""), "tail": (0, "done\n", "")})
    r = s.execute("read_job_log", {"job_id": "60340010", "stream": "stderr", "tail_lines": 1})
    assert r["file"] == "/w/repo/slurm/logs/dk-export-60340010.out" and r["lines"] == ["done"]
    assert "same file" in r["note"]


def test_read_job_log_patterns_for_array_tasks_relative_paths_and_defaults():
    row = {"JobName": "sweep", "User": "u1", "JobIDRaw": "4249"}
    assert slurm_live._expand("logs/%x_%A_%a_%j_%u_%%.out", "4242_7", row) == "logs/sweep_4242_7_4249_u1_%.out"
    assert slurm_live._expand("out-%5j.log", "42", {}) == "out-42.log"
    s, _ = live({"scontrol&show&job": (1, "", ""), "sacct&StdOut": (0, "logs/%j.out|logs/%j.err|/w|j|u|42\n", "")})
    assert s._log_path("42", "stderr") == "/w/logs/42.err"
    s, _ = live({"scontrol&show&job": (1, "", ""), "sacct&StdOut": (0, "||/w|j|u|42\n", "")})
    assert s._log_path("42", "stdout") == "/w/slurm-42.out"


def test_gpu_availability_counts_free_gpus_on_usable_nodes():
    out = ("lrdn0001 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:4(IDX:0-3) mixed\n"
           "lrdn0002 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:1(IDX:0) mixed\n"
           "lrdn0003 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:0(IDX:N/A) idle\n"
           "lrdn0004 boost_usr_prod gpu:a100:4(S:0-1) gpu:a100:0(IDX:N/A) drained\n")
    s, _ = live({"sinfo": (0, out, "")})
    assert s.execute("gpu_availability", {}) == {"partitions": [
        {"partition": "boost_usr_prod", "gpu_type": "a100", "gpus_total": 16, "gpus_free": 7, "idle_nodes": 1}]}


def test_invalid_calls():
    s, _ = live({})
    with pytest.raises(ToolError):
        s.execute("rm_rf", {})
    with pytest.raises(ToolError):
        s.execute("job_status", {})
    assert "invalid job id" in s.execute("job_status", {"job_id": "1; rm -rf ~"})["error"]


def test_remote_setup_is_prepended_to_ssh_commands():
    fake = FakeSlurm({"squeue": (0, "42|train|RUNNING|boost_usr_prod|1:02:03|4:00:00|None\n", "")})
    s = SlurmLive(ssh="c1", remote_setup="module load slurm", runner=fake)
    assert s.execute("list_queue", {"state": "RUNNING"})["jobs"][0]["job_id"] == "42"
    assert fake.calls[0][0] == ["ssh", "-o", "BatchMode=yes", "c1", "module load slurm && " + shlex.join(QUEUE_ARGV)]


def test_without_remote_setup_the_ssh_command_stays_as_before():
    fake = FakeSlurm({"squeue": (0, "42|train|RUNNING|boost_usr_prod|1:02:03|4:00:00|None\n", "")})
    s = SlurmLive(ssh="c1", runner=fake)
    s.execute("list_queue", {"state": "RUNNING"})
    assert fake.calls[0][0] == ["ssh", "-o", "BatchMode=yes", "c1", shlex.join(QUEUE_ARGV)]


def test_remote_setup_is_ignored_for_local_runs():
    fake = FakeSlurm({"squeue": (0, "42|train|RUNNING|boost_usr_prod|1:02:03|4:00:00|None\n", "")})
    s = SlurmLive(remote_setup="module load slurm", runner=fake)
    s.execute("list_queue", {"state": "RUNNING"})
    assert fake.calls[0][0] == QUEUE_ARGV


def test_job_usage_argv_rows_and_error_for_non_running_jobs():
    s, fake = live({"sstat&MaxVMSize": (0, "9|01:30:00|3145728K|1048576K|4194304K\n9.ext|01:00:00|||2097152K\n", "")})
    r = s.execute("job_usage", {"job_id": "9"})
    assert fake.calls[0][0] == ["sstat", "-j", "9", "--format=JobID,AveCPU,MaxRSS,AveRSS,MaxVMSize", "-P", "-n"]
    assert r["steps"][0] == {"JobID": "9", "AveCPU": "01:30:00", "MaxRSS": "3145728K", "AveRSS": "1048576K",
                             "MaxVMSize": "4194304K"} and r["steps"][1]["JobID"] == "9.ext"
    s, _ = live({"sstat": (1, "", "error: Pending job")})
    r = s.execute("job_usage", {"job_id": "9"})
    assert "sstat has no live usage" in r["error"] and "job_accounting" in r["error"]


def test_job_priority_sprio_and_sshare_outputs_with_exit_codes():
    s, fake = live({"sprio&-j&9": (0, "JOBID PARTITION PRIORITY AGE FAIRSHARE\n9 p 100 90 10\n", ""),
                    "sshare&-U&-l": (0, "ACCOUNT USER SHARE LEVELS\nuser 100\n", "")})
    r = s.execute("job_priority", {"job_id": "9"})
    assert [c[0] for c in fake.calls] == [["sprio", "-j", "9", "-l"], ["sshare", "-U", "-l"]]
    assert "9 p 100" in r["priority"] and r["sprio_exit_code"] == 0 and "user 100" in r["shares"]
    assert "note" not in r
    s, _ = live({"sprio&-j&9": (0, "JOBID PARTITION PRIORITY AGE FAIRSHARE\n", "")})
    r = s.execute("job_priority", {"job_id": "9"})
    assert "not pending" in r["note"]


def test_update_job_validates_the_value_and_uses_scontrol_update():
    s, fake = live({"squeue&--me&-j": (0, "9\n", ""), "scontrol&update": (0, "", "")})
    assert s.execute("update_job", {"job_id": "9", "field": "TimeLimit", "value": "2:00:00"}) == {
        "updated": True, "job_id": "9", "field": "TimeLimit", "value": "2:00:00"}
    assert fake.calls[-1][0] == ["scontrol", "update", "JobId=9", "TimeLimit=2:00:00"]
    assert "invalid value" in s.execute("update_job", {"job_id": "9", "field": "TimeLimit", "value": "value 2"})["error"]


def test_hold_release_and_requeue_use_scontrol_on_own_jobs():
    s, fake = live({"squeue&--me&-j": (0, "9\n", ""), "scontrol&hold": (0, "", ""),
                    "scontrol&release": (0, "", ""), "scontrol&requeue": (0, "", "")})
    assert s.execute("hold_job", {"job_id": "9"}) == {"hold": True, "job_id": "9", "message": "Job 9 held"}
    assert s.execute("release_job", {"job_id": "9"}) == {"release": True, "job_id": "9", "message": "Job 9 released"}
    assert s.execute("requeue_job", {"job_id": "9"}) == {"requeue": True, "job_id": "9", "message": "Job 9 requeued"}
    arc = [[c[0][0], c[0][1]] for c in fake.calls if c[0][0] == "scontrol"]
    assert ["scontrol", "hold"] in arc and ["scontrol", "release"] in arc and ["scontrol", "requeue"] in arc
    s, _ = live({"squeue&--me": (0, "", "")})  # not one of the user's own jobs: refused before any scontrol
    s2fake = s._runner
    r = s.execute("hold_job", {"job_id": "9"})
    assert r == {"hold": False, "message": "job 9 is not one of your queued or running jobs"}
    assert not any(c[0][0] == "scontrol" for c in s2fake.calls)


@pytest.mark.parametrize(("command", "ok"), [("squeue --me", True), ("scontrol show job 123", True),
                                             ("rm -rf /", False), ("scontrol update JobId=1 TimeLimit=1", False),
                                             ("squeue | grep x", False), ("sinfo; ls", False)])
def test_run_readonly_command_allowlist(command, ok):
    s, fake = live({"squeue&--me": (0, "1 job\n", ""), "scontrol&show": (0, "JobId=123\n", "")})
    r = s.execute("run_readonly_command", {"command": command})
    if ok:
        assert "error" not in r and r["exit_code"] == 0 and r["command"] == command
        assert fake.calls[-1][0] == shlex.split(command)
    else:
        assert "refused" in r["error"]


def test_run_readonly_command_truncates_long_output():
    s, _ = live({"squeue": (0, "x" * 9000 + "\n", "")})
    r = s.execute("run_readonly_command", {"command": "squeue --me"})
    assert len(r["stdout"]) == 8011 and r["stdout"].endswith("[truncated]")


def test_new_actions_call_confirm_and_return_the_decline_error():
    for name, args in [("update_job", {"job_id": "9", "field": "TimeLimit", "value": "2:00:00"}),
                       ("hold_job", {"job_id": "9"}), ("release_job", {"job_id": "9"}),
                       ("requeue_job", {"job_id": "9"})]:
        s, fake = live({}, confirm=lambda n, a: False)
        r = s.execute(name, args)
        assert r["error"] == "the user declined this action" and r["action"] == name
        assert not fake.calls


def test_round_limit_appends_a_stop_notice_with_the_last_call(monkeypatch, tmp_path):
    s, _ = live({})
    bot = assistant.Assistant(tmp_path, docs=False, slurm=s)
    def stream(messages):  # always wants another tool round, never a final answer
        yield {'tool_calls': [{'name': 'hold_job', 'arguments': '{"job_id": "9"}'}]}
    monkeypatch.setattr(bot, "_stream", stream)
    out = StringIO()
    answer, _ = bot.ask("hold job 9", out=out, max_tool_rounds=1)
    assert "stopped after 1 tool rounds" in out.getvalue() and "hold_job" in out.getvalue()
    notice = bot.history[-1]
    assert notice["role"] == "assistant" and "hold_job" in notice["content"] and '"job_id": "9"' in notice["content"]


def test_agent_loop_runs_slurm_tools_and_returns_results(monkeypatch, tmp_path):
    s, _ = live({"squeue&--me": (0, "77|train|RUNNING|p|0:10|1:00:00|None\n", "")})
    bot = assistant.Assistant(tmp_path, docs=False, slurm=s)
    turns = iter([[{"tool_calls": [{"name": "list_queue", "arguments": "{}"}]}], ["You have one running job, 77."]])
    seen = []
    monkeypatch.setattr(bot, "_stream", lambda messages: (seen.append(list(messages)), next(turns))[1])
    answer, _ = bot.ask("what is running?", out=StringIO())
    assert answer == "You have one running job, 77."
    tool_msg = seen[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["jobs"][0]["job_id"] == "77"


def test_openai_history_links_tool_calls_and_results(monkeypatch, tmp_path):
    """vLLM rejects (HTTP 400) a follow-up request whose tool calls lack ids or whose results lack tool_call_id."""
    s, _ = live({"squeue&--me": (0, "", ""), "sinfo": (0, "", "")})
    bot = assistant.Assistant(tmp_path, docs=False, slurm=s, openai="http://127.0.0.1:1/v1")
    turns = iter([[{"tool_calls": [{"id": "chatcmpl-tool-1", "name": "list_queue", "arguments": "{}"},
                                   {"name": "gpu_availability", "arguments": "{}"}]}], ["done"]])
    seen = []
    monkeypatch.setattr(bot, "_stream", lambda messages: (seen.append(json.loads(json.dumps(messages))), next(turns))[1])
    bot.ask("queue and gpus?", out=StringIO())
    call_msg, r1, r2 = seen[1][-3:]
    ids = [c["id"] for c in call_msg["tool_calls"]]
    assert ids[0] == "chatcmpl-tool-1" and ids[1] and len(set(ids)) == 2
    assert all(isinstance(c["function"]["arguments"], str) for c in call_msg["tool_calls"])
    assert [r1["tool_call_id"], r2["tool_call_id"]] == ids and r1["role"] == r2["role"] == "tool"
