"""Cheap, deterministic answer checks shared by `verify` (training data) and `evaluate` (student outputs)."""

import json
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


def load_commands(path: str | Path | None = None) -> dict:
    """Per-command option lists, scontrol subcommands and sacct/squeue format fields, collected from the commands'
    own --help/--helpformat output on Leonardo (Slurm 25.05, Open MPI 4.1.6); see the file's "source"."""
    text = (files("distillkit") / "data" / "slurm_commands.json").read_text() if path is None else Path(path).read_text()
    return json.loads(text)


def code_blocks(text: str) -> list[tuple[str, str]]:
    return [(lang.lower(), body) for lang, body in CODE_BLOCK.findall(text)]


def _long_opts(tokens: list[str], single_dash_long: bool = False) -> list[str]:
    """Leading options of a Slurm command, stopping at the program being launched. With single_dash_long (Open MPI's
    mpirun) a word like -np or -gpu-bind is a long option taking a value, not -n with "p" attached."""
    opts, expect_value = [], False
    for tok in tokens:
        if tok == "--":  # end of options: what follows is the program
            break
        if tok.startswith("--"):
            opts.append(tok[2:].split("=", 1)[0])
            expect_value = "=" not in tok
        elif tok.startswith("-") and single_dash_long and len(tok) > 2:
            opts.append(tok[1:].split("=", 1)[0])
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


_OTHER_SCHEDULER = re.compile(r"^\s*#(PBS|BSUB)\b", re.MULTILINE)
_FORMAT_OPTS = {"sacct": ("--format", "-o"), "squeue": ("--Format", "-O")}  # field names
_JOB_COMMANDS = ("sbatch", "srun", "salloc")  # also take the sites' plugin options
_SQUEUE_CODES = ("--format", "-o")  # squeue's %-codes


def _option_value(tokens: list[str], names: tuple[str, ...]) -> str | None:
    for i, tok in enumerate(tokens):
        for name in names:
            if tok.startswith(name + "=") and name.startswith("--"):
                return tok.split("=", 1)[1]
            if tok == name and i + 1 < len(tokens):
                return tokens[i + 1]
    return None


def _known(name: str, names) -> bool:
    """Slurm and Open MPI accept a shortened long option (--part for --partition), and sacct/squeue a shortened field
    name (NCPU for NCPUS), so any prefix of a real name counts as known."""
    return any(n.startswith(name) for n in names)


def command_problems(text: str, spec: dict) -> list[str]:
    """Options, subcommands and format fields an answer uses that the commands do not have (issue #110): long options
    of every Slurm command and mpirun, #SBATCH directives, scontrol subcommands, sacct/squeue format fields and
    squeue %-codes, and #PBS/#BSUB directives in a Slurm answer."""
    problems = set()
    plugins = spec["site_plugin_options"]["options"]
    options = {c: [*o, *plugins] if c in _JOB_COMMANDS else o for c, o in spec["options"].items()}
    if m := _OTHER_SCHEDULER.search(text):
        problems.add(f"#{m.group(1)} directive (not Slurm)")
    for directive in SBATCH_LINE.findall(text):
        problems |= {f"sbatch: no option --{o}" for o in _long_opts(directive.split())
                     if not _known(o, options["sbatch"])}
    for _, body in code_blocks(text):
        for line in body.splitlines():
            if line.lstrip().startswith("#"):
                continue
            line = re.sub(r"\$\(|[();|&]", " ", line)
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                tokens = line.split()
            for i, tok in enumerate(tokens):
                cmd, rest = tok.rsplit("/", 1)[-1], tokens[i + 1:]
                # a launcher's own options end where the command it starts begins (srun ... mpirun --x)
                end = next((j for j, t in enumerate(rest) if t.rsplit("/", 1)[-1] in spec["options"] or t == "scontrol"), None)
                rest = rest[:end]
                if cmd == "scontrol":
                    sub = next((t for t in rest if not t.startswith("-")), None)
                    if sub and not PLACEHOLDER.fullmatch(sub) and sub.lower() not in spec["scontrol_commands"]:
                        problems.add(f"scontrol: no command {sub!r}")
                    continue
                if cmd not in spec["options"]:
                    continue
                problems |= {f"{cmd}: no option --{o}" for o in _long_opts(rest, cmd == "mpirun")
                             if not _known(o, options[cmd])}
                if cmd in _FORMAT_OPTS and (value := _option_value(rest, _FORMAT_OPTS[cmd])):
                    known = {f.lower() for f in spec[f"{cmd}_fields"]}
                    for field in filter(None, (f.split("%")[0].split(":")[0].strip() for f in value.split(","))):
                        if not _known(field.lower(), known) and not PLACEHOLDER.fullmatch(field):
                            problems.add(f"{cmd}: no format field {field!r}")
                if cmd == "squeue" and (value := _option_value(rest, _SQUEUE_CODES)):
                    problems |= {f"squeue: no format code %{c}" for c in re.findall(r"%[.\-]?\d*([A-Za-z])", value)
                                 if c not in spec["squeue_codes"]}
    return sorted(problems)


def bash_syntax_ok(script: str) -> bool:
    # doc-style placeholders like `<jobid>` would parse as redirections
    script = PLACEHOLDER.sub("PLACEHOLDER", script)
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash not found on PATH: the shell-syntax check needs it")
    r = subprocess.run([bash, "-n"], input=script, text=True, errors="replace", capture_output=True)
    return r.returncode == 0


def check_answer(answer: str, valid_flags: set[str], commands: dict | None = None) -> dict:
    """bad_flags/bash_ok/passed keep their meaning (the mock scheduler relies on them); with `commands` (see
    load_commands) the stricter per-command check is added as command_problems."""
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
        **({"command_problems": command_problems(answer, commands)} if commands is not None else {}),
    }
