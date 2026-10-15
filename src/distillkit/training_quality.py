"""Bounded command/intent checks on training targets, before model loading.

These checks quarantine suspicious targets for review. They neither repair
answers nor certify the remaining records as semantically correct.
"""

import json
import re
from pathlib import Path

from .records import final_answer


def command_target_findings(question: str, answer: str) -> list[str]:
    blocks = re.findall(r'```(?:bash|sh|shell|zsh)?\s*\n(.*?)```', answer, re.S)
    # Check commands, not comments or quoted prose. This deliberately does not
    # claim to parse arbitrary shell programs, short options or inline examples.
    code = '\n'.join(line for block in blocks for line in block.splitlines()
                     if not line.lstrip().startswith('#')).replace('\\\n', ' ')
    findings = []
    if re.search(r'\b(?:srun|sbatch|salloc)\b[^\n]*--distribution(?:=|\s+)plane,\d', code):
        findings.append('invalid_plane_distribution_separator')
    if (re.search(r'\b(?:standard output|stdout)\b', question, re.I)
            and re.search(r'\bsattach\b', code)
            and '--error-filter' in code and '--output-filter' not in code):
        findings.append('stdout_requirement_missing_output_filter')
    return findings


def require_training_target_quality(rows: list[dict], evidence_dir: Path | None = None) -> dict:
    findings = []
    for row in rows:
        codes = command_target_findings(row.get('question', ''), final_answer(row))
        if codes:
            findings.append({'id': row.get('id', '<missing>'), 'codes': codes})
    report = {'protocol': 'bounded-command-intent-v1', 'rows': len(rows),
              'findings': findings, 'passed': not findings,
              'semantic_certification': False, 'answers_modified': False}
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / 'training_target_quality.json').write_text(json.dumps(report, indent=2) + '\n')
    if findings:
        raise ValueError(f'{len(findings)} training targets need command/intent review '
                         f'(examples: {findings[:5]}). Quarantine or correct and re-review '
                         'them before creating fresh training inputs; do not edit frozen runs.')
    return report
