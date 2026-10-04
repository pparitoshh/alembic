"""Cheap, deterministic answer checks shared by `verify` (training data) and `evaluate` (student outputs)."""

import re
import shlex
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

CODE_BLOCK = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)
SBATCH_LINE = re.compile(r"^\s*#SBATCH\s+(.*)$", re.MULTILINE)
SLURM_CMDS = {"sbatch", "srun", "salloc"}
PLACEHOLDER = re.compile(r"<[A-Za-z_][\w.-]*>")


def load_flags(path: str | Path | None = None) -> set[str]:
    """Valid Slurm long options, one per line. None = the list shipped in the package (Slurm man pages),
    so an installed copy (e.g. on Leonardo) finds it without the repo checkout."""
    text = (files("distillkit") / "data" / "slurm_flags.txt").read_text() if path is None else Path(path).read_text()
    lines = text.splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


def code_blocks(text: str) -> list[tuple[str, str]]:
    return [(lang.lower(), body) for lang, body in CODE_BLOCK.findall(text)]


def _long_opts(tokens: list[str]) -> list[str]:
    """Leading options of a Slurm command, stopping at the program being launched."""
    opts, expect_value = [], False
    for tok in tokens:
        if tok == "--":  # end of options: what follows is the program
            break
        if tok.startswith("--"):
            opts.append(tok[2:].split("=", 1)[0])
            expect_value = "=" not in tok
        elif tok.startswith("-"):
            expect_value = len(tok) == 2
        elif expect_value:
            expect_value = False
        else:
            break
    return opts


def slurm_flags_used(text: str) -> list[str]:
    """Long options used in #SBATCH directives and sbatch/srun/salloc command lines."""
    flags = []
    for directive in SBATCH_LINE.findall(text):
        flags += _long_opts(directive.split())
    for _, body in code_blocks(text):
        for line in body.splitlines():
            if line.lstrip().startswith("#"):
                continue
            # split command substitutions / chains so `$(sbatch ...)` and `a; sbatch ...` are seen
            line = re.sub(r"\$\(|[();|&]", " ", line)
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                tokens = line.split()
            for i, tok in enumerate(tokens):
                if tok in SLURM_CMDS:
                    flags += _long_opts(tokens[i + 1 :])
    return flags


def bash_syntax_ok(script: str) -> bool:
    # doc-style placeholders like `<jobid>` would parse as redirections
    script = PLACEHOLDER.sub("PLACEHOLDER", script)
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash not found on PATH: the shell-syntax check needs it")
    r = subprocess.run([bash, "-n"], input=script, text=True, capture_output=True)
    return r.returncode == 0


def check_answer(answer: str, valid_flags: set[str]) -> dict:
    flags = slurm_flags_used(answer)
    bad = sorted({f for f in flags if f not in valid_flags})
    shell_blocks = [b for lang, b in code_blocks(answer) if lang in ("bash", "sh", "shell", "")]
    syntax_ok = all(bash_syntax_ok(b) for b in shell_blocks)
    return {
        "n_flags": len(flags),
        "bad_flags": bad,
        "n_shell_blocks": len(shell_blocks),
        "bash_ok": syntax_ok,
        "passed": not bad and syntax_ok,
    }
