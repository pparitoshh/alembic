"""Check explicit duration comments on single-option SBATCH time directives.

This is a conservative contradiction check, not a command validity checker.
Relative comments, free prose, multiple options, variables and shell commands
are deliberately outside its scope. Slurm rounds seconds up to whole minutes;
zero requests an unlimited duration. Reference:
https://slurm.schedmd.com/sbatch.html#OPT_time
"""
from decimal import Decimal, ROUND_CEILING
import re

_LITERAL = r'\d+(?:-\d+)?(?::\d+){0,2}'
_DIRECTIVE = re.compile(
    r'^\s*#SBATCH\s+(?:--time(?:=|\s+)|-t\s*)'
    r'(?P<quote>[\'\"]?)(?P<value>' + _LITERAL + r')(?P=quote)\s+#\s*(?P<comment>.*)$'
)
_COUNT = r'\d+(?:\.\d+)?\s*(?:days?|hours?|hrs?|minutes?|mins?|seconds?|secs?)'
_DURATION_COMMENT = re.compile(
    r'(?:(?:time limit|walltime|timeout|time)\s*:?\s*)?'
    r'(?P<duration>' + _COUNT + r'(?:(?:\s*,?\s+|\s+and\s+)' + _COUNT + r')*)'
    r'(?:\s+(?:time limit|walltime|timeout))?\s*[.!]?', re.I
)
_FORMAT_COMMENT = re.compile(r'(?:(?:timeout|time|walltime)\s+(?:in\s+)?)?HH:MM', re.I)


def duration_minutes(value: str) -> int | None:
    """Parse the six documented literal forms, rounding seconds upward."""
    if len(value) > 64 or re.fullmatch(_LITERAL, value) is None:
        raise ValueError('Not a supported literal Slurm time')
    days = 0
    if '-' in value:
        day, value = value.split('-')
        days = int(day)
        factors = (3600, 60, 1)
    else:
        factors = {1: (60,), 2: (60, 1), 3: (3600, 60, 1)}[len(value.split(':'))]
    seconds = days * 86400 + sum(int(part) * factor for part, factor in zip(value.split(':'), factors))
    return (seconds + 59) // 60 if seconds else None


def check_time_comments(text: str) -> list[dict]:
    issues = []
    for line in text.splitlines():
        match = _DIRECTIVE.fullmatch(line)
        if match is None:
            continue
        value, comment = match['value'], match['comment'].strip()
        if len(value) > 64:
            continue
        effective = duration_minutes(value)
        duration = _DURATION_COMMENT.fullmatch(comment)
        reason = None
        if duration:
            seconds = sum(
                Decimal(number) * {'d': 86400, 'h': 3600, 'm': 60, 's': 1}[unit[0].lower()]
                for number, unit in re.findall(r'(\d+(?:\.\d+)?)\s*([a-z]+)', duration['duration'], re.I)
            )
            claimed = int((seconds / 60).to_integral_value(rounding=ROUND_CEILING)) if seconds else None
            if effective != claimed:
                reason = 'The explicit duration comment disagrees with the Slurm time value after minute rounding'
        elif '-' not in value and value.count(':') == 1 and _FORMAT_COMMENT.fullmatch(comment):
            reason = 'A two-field Slurm time without a day prefix means minutes:seconds, not HH:MM'
        if reason:
            issues.append(dict(option='time', value=value, reason=reason,
                               effective_minutes=effective, comment=comment))
    return issues
