"""Conservative checks for known invalid literal Slurm option values.

This is not a full shell or Slurm parser. Only SBATCH directives and direct
Slurm invocations in shell blocks are inspected. Unknown option arities stop
command scanning so an application's arguments are never assumed to be Slurm
options. Variables, templates, and unhandled value forms remain unchecked.
Reference: https://slurm.schedmd.com/sbatch.html#OPT_distribution
"""
import re
import shlex
from pathlib import PurePosixPath

PROTOCOL = 'slurm-known-invalid-values-v2'
_COMMANDS = {'sbatch', 'salloc', 'srun'}
_BOOLEANS = {'--exclusive', '--overcommit', '--oversubscribe', '--parsable',
             '--test-only', '--pty', '--label', '--verbose', '--quiet'}
_VALUES = {'--nodes', '--ntasks', '--ntasks-per-node', '--cpus-per-task', '--mem',
           '--mem-per-cpu', '--partition', '--account', '--qos', '--time',
           '--gres', '--gpus', '--gpus-per-node', '--constraint', '--job-name',
           '--output', '--error', '--chdir', '--dependency', '--export',
           '-N', '-n', '-c', '-p', '-A', '-t', '-J', '-o', '-e', '-D'}


def distribution_issue(value: str) -> str | None:
    """Reject known slot/separator mistakes, without certifying other forms.

The documented grammar and Slurm 25.05.7 native parser agree on these errors.
In particular, empty/default fields and plane sizes can depend on context;
we do not reject them just because they are outside this narrow check.
"""
    if not re.fullmatch(r'[A-Za-z0-9_*,=:.-]+', value):
        return None
    levels = value.lower().split(',', 1)[0].split(':')
    if levels[0] == 'fcyclic':
        return 'fcyclic applies to socket/core distribution, not node distribution'
    if any(level in {'pack', 'nopack'} for level in levels[1:]):
        return 'Pack/NoPack follows a comma, not a colon-separated CPU distribution field'
    if any(level == 'arbitrary' for level in levels[1:]):
        return 'arbitrary applies to node distribution, not socket/core distribution'
    if ',' in value and value.lower().split(',', 1)[1] == 'fcyclic':
        return 'fcyclic belongs in a colon-separated CPU distribution field'
    return None


def _distribution_values(tokens: list[str], *, directive: bool):
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == '--' or tok in {';', '&&', '||', '|', '&'}:
            break
        if tok.startswith('--distribution='):
            yield tok.split('=', 1)[1]
        elif tok in {'--distribution', '-m'}:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith('-'):
                break
            i += 1
            yield tokens[i]
        elif tok.startswith('-m') and not tok.startswith('--') and len(tok) > 2:
            yield tok[2:]
        elif not directive:
            if tok in _VALUES:
                i += 1
            elif tok in _BOOLEANS or (tok.startswith('--') and '=' in tok):
                pass
            elif len(tok) > 2 and tok[:2] in _VALUES and not tok.startswith('--'):
                pass
            else:
                break  # executable, payload, or an option with unknown arity
        i += 1


def check_distribution_values(text: str) -> list[dict]:
    from .checks import code_blocks
    sources = [(match.group(1), True) for match in re.finditer(r'^\s*#SBATCH\s+(.*)$', text, re.M)]
    for language, body in code_blocks(text):
        if language not in {'', 'bash', 'sh', 'shell'}:
            continue
        for line in body.replace('\\\n', ' ').splitlines():
            if not line.lstrip().startswith('#'):
                sources.append((line, False))
    issues = []
    for line, directive in sources:
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=';&|')
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            continue  # shell syntax is checked separately
        if not directive:
            if tokens[:1] == ['$']:
                tokens = tokens[1:]
            if not tokens or PurePosixPath(tokens[0]).name not in _COMMANDS:
                continue
            tokens = tokens[1:]
        for value in _distribution_values(tokens, directive=directive):
            issue = distribution_issue(value)
            if issue:
                issues.append({'option': 'distribution', 'value': value, 'reason': issue})
    return issues
