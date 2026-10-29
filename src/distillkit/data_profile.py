"""CPU-only dataset profile and optional training suitability gate.

Keyword matches are discovery aids, never semantic relevance labels. This gate
supplements existing verification, toolcheck, decontamination and provenance;
it does not replace them. Domain review is an explicitly trusted input.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import TrainingDataRequirementsCfg, TrainingInputManifestCfg
from .records import final_answer


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def records_sha256(rows: list[dict]) -> str:
    """Ordered parsed-record fingerprint; independent of JSON whitespace."""
    return sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'),
                            ensure_ascii=False, allow_nan=False).encode())


def profile(rows: list[dict]) -> dict:
    modes = Counter(r.get('mode', 'prose') for r in rows)
    calls = sum(any(m.get('role') == 'assistant' and m.get('tool_calls')
                    for m in r.get('messages', [])) for r in rows)
    tool_rows = sum(modes[m] for m in ('call', 'ask', 'none'))
    commands = re.compile(r'\b(?:slurm|sbatch|squeue|sacct|srun|scancel|scontrol|salloc)\b', re.I)
    questions = [r.get('question', '') for r in rows]
    answers = [final_answer(r) for r in rows]
    normalized = [' '.join(unicodedata.normalize('NFKC', q).casefold().split()) for q in questions]
    invalid = []
    for r in rows:
        mode = r.get('mode', 'prose')
        has_call = any(m.get('role') == 'assistant' and m.get('tool_calls') for m in r.get('messages', []))
        if (mode not in ('prose', 'call', 'ask', 'none')
                or (mode != 'prose' and not r.get('tools'))
                or (mode == 'call' and not has_call)
                or (mode != 'call' and has_call)):
            invalid.append(r.get('id'))
    return {
        'schema': 'dataset-profile-v1', 'rows': len(rows),
        'records_sha256': records_sha256(rows),
        'unique_ids': len({r.get('id') for r in rows}),
        'missing_ids': sum(not r.get('id') for r in rows),
        'unique_documents': len({r.get('doc_id') for r in rows if r.get('doc_id')}),
        'missing_documents': sum(not r.get('doc_id') for r in rows),
        'duplicate_normalized_questions': len(normalized) - len(set(normalized)),
        'empty_questions': sum(not q.strip() for q in questions),
        'empty_final_answers': sum(not a.strip() for a in answers),
        'modes': dict(sorted(modes.items())),
        'tool_decision_rows': tool_rows,
        'tool_decision_fraction': tool_rows / len(rows) if rows else 0,
        'actual_tool_call_rows': calls,
        'mode_structure_errors': invalid,
        'tasks': dict(sorted(Counter(r.get('task', '<missing>') for r in rows).items())),
        'personas': dict(sorted(Counter(r.get('persona', '<missing>') for r in rows).items())),
        'source_families': dict(sorted(Counter(r.get('source_family') or '<unrecorded>' for r in rows).items())),
        'median_final_answer_characters': median(map(len, answers)) if rows else 0,
        'slurm_command_keyword_question_rows': sum(bool(commands.search(q)) for q in questions),
        'slurm_command_keyword_question_or_answer_rows': sum(bool(commands.search(q + '\n' + a)) for q, a in zip(questions, answers)),
        'limitations': [
            'Keyword matches are not a count of useful or correct HPC examples.',
            'Source families and personas are recorded metadata, not independent relevance judgments.',
            'Tool modes include calls, clarifications and deliberate no-call decisions; structure checks are not tool validation.',
            'No semantic deduplication, source-grounding, licensing or evaluation-leakage certification is performed here.',
        ],
    }


class DomainReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    version: Literal['training-domain-review-v1']
    records_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    decision: Literal['approved-for-hpc-training']
    approved_record_ids: list[str] = Field(min_length=1)
    review_method: str = Field(min_length=1)
    # Supporting row/source relevance judgments, relative to the review JSON.
    evidence: TrainingInputManifestCfg


def _bound_bytes(binding: TrainingInputManifestCfg, root: Path = Path('.')) -> bytes:
    raw = (root / binding.path).read_bytes()
    if sha256(raw) != binding.sha256:
        raise ValueError(f'hash mismatch: {binding.path}')
    return raw


def enforce_requirements(rows: list[dict], requirements: TrainingDataRequirementsCfg,
                         *, evidence_dir: Path | None = None) -> dict:
    report = profile(rows)
    failures = []
    if not rows or report['missing_ids'] or report['unique_ids'] != len(rows):
        failures.append('records must have nonempty unique IDs')
    if report['missing_documents'] or report['empty_questions'] or report['empty_final_answers']:
        failures.append('missing documents, questions or final answers')
    if report['duplicate_normalized_questions']:
        failures.append('duplicate normalized questions')
    if report['mode_structure_errors']:
        failures.append('tool mode/structure mismatch')
    if not requirements.min_tool_fraction <= report['tool_decision_fraction'] <= requirements.max_tool_fraction:
        failures.append('tool-decision fraction outside configured bounds')
    if any(not report['modes'].get(mode) for mode in requirements.required_modes):
        failures.append('missing required tool decision mode')
    if any(not report['tasks'].get(task) for task in requirements.required_tasks):
        failures.append('missing required task type')
    try:
        review = DomainReview.model_validate_json(_bound_bytes(requirements.domain_review))
        if review.records_sha256 != report['records_sha256']:
            raise ValueError('domain review refers to different records')
        if (len(set(review.approved_record_ids)) != len(review.approved_record_ids)
                or set(review.approved_record_ids) != {r.get('id') for r in rows}):
            raise ValueError('domain approval does not cover exactly the selected record IDs')
        _bound_bytes(review.evidence, requirements.domain_review.path.parent)
    except (OSError, ValueError) as exc:
        failures.append(f'domain review: {exc}')
    report.update(status='FAIL' if failures else 'PASS', failures=failures,
                  requirements=requirements.model_dump(mode='json'))
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / 'training_data_profile.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        raise ValueError('training data suitability: ' + '; '.join(failures))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.records.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    report = profile(rows)
    report['file_sha256'] = sha256(raw)
    report['file'] = str(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Profiled {len(rows)} rows; this is not a relevance certificate. {args.output}")


if __name__ == '__main__':
    main()
