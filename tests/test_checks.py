from distillkit.checks import check_answer, slurm_flags_used
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
