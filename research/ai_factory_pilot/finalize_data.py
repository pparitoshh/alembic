"""Fail-closed handoff from reviewed generation to a small training experiment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

from distillkit.config import load_config
from distillkit.heldout import EVAL_ROOT, heldout_questions
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import final_answer, training_example
from distillkit.source_resolver import SourceResolver
from distillkit.train import token_lengths
from distillkit.verify import _shingles, _jaccard

cfg = load_config(sys.argv[1])
root = cfg.run_dir
rows = read_jsonl(root / 'clean.jsonl')
held = heldout_questions(EVAL_ROOT)
resolver = SourceResolver(cfg)
accepted, rejected, seen = [], [], []
for row in rows:
    resolver.check_record(row)
    if row.get('grounding_checks', {}).get('status') != 'supported':
        raise RuntimeError('Missing positive source-grounding review: ' + row['id'])
    # Attached source blocks must not hide duplicated/held-out question wording.
    raw = row.get('question_generation', {}).get('raw_question', row['question'])
    sh = _shingles(raw)
    reason = None
    if any(_jaccard(sh, h) >= 0.8 for _, _, h in held):
        reason = 'raw_question_eval_overlap'
    elif any(_jaccard(sh, prior) >= 0.8 for prior in seen):
        reason = 'raw_question_duplicate'
    elif not final_answer(row).strip():
        reason = 'missing_final_answer'
    if reason:
        rejected.append({'id': row['id'], 'reason': reason})
    else:
        accepted.append(row)
        seen.append(sh)
resolver.assert_unchanged()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(cfg.student.model, local_files_only=True)
examples = [training_example(r, cfg.student.system_prompt) for r in accepted]
lengths = token_lengths(tok, examples)
eligible = []
for row, length in zip(accepted, lengths, strict=True):
    if length > cfg.train.max_length:
        rejected.append({'id': row['id'], 'reason': 'would_truncate', 'tokens': length})
    else:
        eligible.append(row)
prose = [r for r in eligible if r.get('mode') == 'prose']
tools = [r for r in eligible if r.get('mode') != 'prose']
random.Random(42).shuffle(prose)
random.Random(43).shuffle(tools)
count_prose = min(len(prose), int(len(tools) * 7 / 3))
count_tools = min(len(tools), round(count_prose * 3 / 7))
selected = prose[:count_prose] + tools[:count_tools]
modes = Counter(r.get('mode') for r in selected)
if len(selected) < 100 or any(modes[m] < 5 for m in ['call', 'ask', 'none']):
    report = {'status': 'insufficient_reviewed_coverage_no_training', 'eligible': len(eligible),
              'balanced_rows': len(selected), 'modes': dict(modes), 'rejections': rejected}
    (root / 'training_gate.json').write_text(json.dumps(report, indent=2))
    raise SystemExit('Training gate stopped: need >=100 rows and >=5 each call/ask/none. No thresholds relaxed.')
random.Random(44).shuffle(selected)
destination = root / 'training'
destination.mkdir(exist_ok=True)
write_jsonl(destination / 'verified.jsonl', selected)
report = {'status': 'passed_for_pilot_only', 'reviewed_clean_rows': len(rows), 'selected_rows': len(selected),
          'modes': dict(modes), 'tool_fraction': count_tools / len(selected),
          'source_documents': len({r['doc_id'] for r in selected}),
          'eval_question_count': len(held), 'raw_question_and_source_checks': True,
          'no_selected_record_truncated': True, 'rejections': rejected,
          'sha256': hashlib.sha256((destination / 'verified.jsonl').read_bytes()).hexdigest(),
          'release_approved': False}
(root / 'training_gate.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
