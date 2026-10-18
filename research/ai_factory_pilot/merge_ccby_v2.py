"""Merge the reviewed CC-BY shards into one fail-closed release. CPU only.

usage: merge_ccby_v1.py SHARDS BANK.jsonl [BANK.jsonl ...]
Banks are existing training rows; a new row whose raw question repeats one is dropped.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from distillkit.config import load_config
from distillkit.heldout import EVAL_ROOT, heldout_questions
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import final_answer, training_example
from distillkit.source_resolver import SourceResolver
from distillkit.train import token_lengths
from distillkit.verify import _shingles, _jaccard

shards, banks = int(sys.argv[1]), sys.argv[2:]
out = Path('runs/ai_factory_ccby_v2/release')
out.mkdir(parents=True, exist_ok=True)


def raw_question(row):
    return row.get('question_generation', {}).get('raw_question', row['question'])


held = heldout_questions(EVAL_ROOT)
bank = [_shingles(raw_question(r)) for b in banks for r in read_jsonl(Path(b))]
accepted, rejected, seen, per_shard = [], [], [], {}
for k in range(shards):
    cfg = load_config('configs/ai_factory_ccby_v2_shard_%d.json' % k)
    path = cfg.run_dir / 'clean.jsonl'
    if not path.exists():
        per_shard[k] = 'missing_clean_output'
        continue
    rows = read_jsonl(path)
    per_shard[k] = len(rows)
    resolver = SourceResolver(cfg)
    for row in rows:
        resolver.check_record(row)
        if row.get('grounding_checks', {}).get('status') != 'supported':
            raise RuntimeError('Missing positive source-grounding review: ' + row['id'])
        sh = _shingles(raw_question(row))
        reason = None
        if any(_jaccard(sh, h) >= 0.8 for _, _, h in held):
            reason = 'raw_question_eval_overlap'
        elif any(_jaccard(sh, prior) >= 0.8 for prior in seen):
            reason = 'raw_question_duplicate'
        elif any(_jaccard(sh, prior) >= 0.8 for prior in bank):
            reason = 'raw_question_in_existing_training_bank'
        elif not final_answer(row).strip():
            reason = 'missing_final_answer'
        if reason:
            rejected.append({'id': row['id'], 'reason': reason})
        else:
            accepted.append(row)
            seen.append(sh)
    resolver.assert_unchanged()

from transformers import AutoTokenizer
cfg = load_config('configs/ai_factory_ccby_v2_shard_0.json')
tok = AutoTokenizer.from_pretrained(cfg.student.model, local_files_only=True)
lengths = token_lengths(tok, [training_example(r, cfg.student.system_prompt) for r in accepted])
release = []
for row, length in zip(accepted, lengths, strict=True):
    if length > cfg.train.max_length:
        rejected.append({'id': row['id'], 'reason': 'would_truncate', 'tokens': length})
    else:
        release.append(row)
write_jsonl(out / 'verified.jsonl', release)
modes = Counter(r.get('mode') for r in release)
report = {
    'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'automated_gates_passed_not_human_reviewed',
    'licence_scope': 'CC-BY-4.0 (LUMI, CSC); attribution in each row source fields',
    'clean_rows_per_shard': per_shard, 'released_rows': len(release), 'modes': dict(modes),
    'tool_fraction': round(sum(v for m, v in modes.items() if m != 'prose') / max(1, len(release)), 3),
    'sites': dict(Counter(r['doc_id'].split('_')[0] for r in release)),
    'source_documents': len({r['doc_id'] for r in release}),
    'rejections': dict(Counter(r['reason'] for r in rejected)), 'bank_rows': len(bank),
    'median_answer_chars': sorted(len(final_answer(r)) for r in release)[len(release) // 2] if release else 0,
    'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'REJECTED.json').write_text(json.dumps(rejected, indent=1))
(out / 'REPORT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
