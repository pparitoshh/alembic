import pytest

from distillkit.checks import check_answer, command_problems, load_commands, slurm_flags_used
from distillkit.seeds import chunk_text

FLAGS = {"time", "mem", "array", "ntasks", "pty", "gpus-per-node", "parsable", "dependency"}


def test_sbatch_directives_and_commands():
    ans = """Try this:
```bash
#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem 4G
srun --ntasks=2 python train.py --epochs 3
jid=$(sbatch --parsable a.sh); sbatch --dependency=afterok:$jid b.sh
```"""
    assert slurm_flags_used(ans) == ["time", "mem", "ntasks", "parsable", "dependency"]
    assert check_answer(ans, FLAGS)["passed"]


def test_double_dash_ends_options():
    ans = "```bash\nsalloc --nodes=1 --time=01:00:00 -- /bin/bash -c 'srun --ntasks=4 hostname'\n```"
    assert slurm_flags_used(ans) == ["nodes", "time"]  # no "" from the bare `--`


def test_hallucinated_flag_is_caught():
    ans = "```bash\n#SBATCH --gpu-count=2\n#SBATCH --time=1:00:00\n```"
    res = check_answer(ans, FLAGS)
    assert res["bad_flags"] == ["gpu-count"]
    assert not res["passed"]


def test_bash_syntax_error_is_caught():
    ans = "```bash\nif [ -f x ]; then\n  echo hi\n```"
    assert not check_answer(ans, FLAGS)["bash_ok"]


def test_placeholders_are_not_syntax_errors():
    ans = "```bash\nscontrol show job <jobid>\nsacct -j <job_id> --format=MaxRSS\n```"
    assert check_answer(ans, FLAGS)["bash_ok"]


def test_chunking_keeps_code_blocks_whole():
    text = "para one\n\n```bash\nline1\n\nline2\n```\n\npara two"
    chunks = chunk_text(text, max_chars=10)
    assert any("line1\n\nline2" in c for c in chunks)


COMMANDS = load_commands()


def _bash(*lines):
    return "```bash\n" + "\n".join(lines) + "\n```"


@pytest.mark.parametrize("line, problem", [  # issue #110 examples
    ("scontrol cancel 123", "scontrol: no command 'cancel'"),
    ("sacct --jobid 123", "sacct: no option --jobid"),
    ("sacct --nodes=lrdn0001", "sacct: no option --nodes"),
    ("sacct -j 123 --format=JobID,MaxGPU", "sacct: no format field 'MaxGPU'"),
    ("sacct -j 123 -o JobID,AllocTime", "sacct: no format field 'AllocTime'"),
    ("squeue -O JobID,GPUUtil", "squeue: no format field 'GPUUtil'"),
    ('squeue -o "%i %Q %s"', "squeue: no format code %s"),  # squeue: Invalid job format specification: s
    ("mpirun -np 8 --gpu-bind per-core ./app", "mpirun: no option --gpu-bind"),
    ("mpirun -np 8 -gpu-bind per-core ./app", "mpirun: no option --gpu-bind"),
    ("srun -n 8 mpirun -np 8 --gpu-bind per-core ./app", "mpirun: no option --gpu-bind"),
    ("srun --gpu-count=1 --pty bash", "srun: no option --gpu-count"),
    ("salloc -N 1 --pty bash", "salloc: no option --pty"),
    ("sacct -j 123 --sort=End", "sacct: no option --sort"),
    ("sbatch --mpi=pmix job.sh", "sbatch: no option --mpi"),  # srun has it, sbatch rejects it
    ("sacct -j 123 -o JobID,AllocGRES", "sacct: no format field 'AllocGRES'"),  # removed, use AllocTRES
    ("#SBATCH --mount /a:/b", "sbatch: no option --mount"),
    ("#PBS -l nodes=2", "#PBS directive (not Slurm)"),
    ("#BSUB -n 4", "#BSUB directive (not Slurm)"),
])
def test_made_up_commands_are_caught(line, problem):
    assert command_problems(_bash(line), COMMANDS) == [problem]


@pytest.mark.parametrize("line", [
    "#SBATCH --gres=gpu:4",
    "srun --gpu-bind=closest --cpu-bind=cores ./app",
    "sacct -j 123 --format=JobID,State,MaxRSS%12,Elapsed,ReqTRES",
    "sacct -X --starttime=now-7days -o jobid,jobname,state",  # field names are case-insensitive
    'squeue --me -o "%.18i %.9P %j %T %R"',  # --me is missing from squeue --help but real
    'squeue -h -o "%i %b %y %Q"',  # %b (GRES) is not in the man page but squeue accepts it
    "squeue -O JobID:10,Partition,State",
    "sbatch --test-only job.sh",  # also missing from sbatch --help
    "srun --interactive --pty bash",  # hidden, but srun --test-only accepts it
    "sbatch --tasks-per-node=4 --part=boost_usr_prod job.sh",  # hidden alias; a shortened option
    "sacct -X --account=myproj -o JobID,NCPU",  # prefixes of --accounts and NCPUS
    "srun --container-image=nvcr.io#nvidia/pytorch:24.05-py3 python t.py",  # pyxis plugin
    "#SBATCH --ear=on\n#SBATCH --ear-policy=monitoring",  # EAR plugin (LRZ, BSC, SURF)
    "srun --environment=pytorch --pty bash",  # CSCS Alps container engine
    "scancel --me --state=PENDING",
    "scontrol show job 123",
    "scontrol update JobId=<jobid> TimeLimit=2:00:00",
    "scontrol requeue 123",
    "mpirun -np 8 -x LD_LIBRARY_PATH --map-by ppr:4:node --bind-to core ./app --epochs 3",
    "mpirun -n 4 python train.py --lr 1",  # the program's own options are not mpirun's
    "watch -n 5 squeue --me",
    "jid=$(sbatch --parsable a.sh); sbatch --dependency=afterok:$jid b.sh",
    "# sacct --jobid in a comment is ignored",
])
def test_real_commands_pass(line):
    assert command_problems(_bash(line), COMMANDS) == []


def test_prose_outside_code_blocks_is_not_checked():
    assert command_problems("Note that `scontrol cancel` does not exist; use scancel.", COMMANDS) == []


def test_check_answer_keeps_its_keys_without_commands():
    ans = _bash("scontrol cancel 123")
    assert "command_problems" not in check_answer(ans, FLAGS)  # the mock scheduler's view is unchanged
    res = check_answer(ans, FLAGS, COMMANDS)
    assert res["passed"] and res["command_problems"] == ["scontrol: no command 'cancel'"]
