"""Preserve accepted original rows; combine with newly reviewed candidates for full re-verification."""
import hashlib
import json
from pathlib import Path
from distillkit.config import load_config
from distillkit.source_resolver import SourceResolver
from distillkit.support import check_support

repo = Path(__file__).resolve().parents[2]
old = repo / 'runs/ai_factory_pilot_20261013'
new = repo / 'runs/tool_recovery_20261014'
out = repo / 'runs/ai_factory_recovery_20261014'
cfg = load_config(repo / 'configs/ai_factory_recovery.json')
resolver = SourceResolver(cfg)

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

selected = rows(old / 'clean.jsonl')
original_rows = {r['id']: r for r in rows(old / 'generated.jsonl')}
original_reviews = {r['id']: r for r in rows(old / 'grounding_reviews.jsonl')}
original_questions = {r['id']: r for r in rows(old / 'questions.jsonl')}
new_rows = rows(new / 'generated.jsonl')
new_reviews = rows(new / 'grounding_reviews.jsonl')
accepted_raw, accepted_reviews, lineage = [], [], []
for row in selected:
    raw = original_rows[row['id']]
    source = resolver.source_for_record(raw)
    question = original_questions[raw['id'].rsplit('/s', 1)[0]]
    assert raw['question'] == question['question'] and raw['chunk_id'] == question['chunk_id']
    assert raw['generation_manifest_sha256'] == json.loads((old / 'generation_manifest.json').read_text())['binding_sha256']
    support = check_support(raw, source, original_reviews[row['id']], protocol=cfg.verify.grounding_review_protocol)
    assert support['status'] == 'supported', 'Original acceptance no longer reproduces: ' + row['id']
    accepted_raw.append(raw)
    accepted_reviews.append(original_reviews[row['id']])
    lineage.append({'id': row['id'], 'origin': str(old), 'original_accepted': True})
for row in new_rows:
    resolver.source_for_record(row)
    assert row['generation_manifest_sha256'] == json.loads((new / 'generation_manifest.json').read_text())['binding_sha256']
    lineage.append({'id': row['id'], 'origin': str(new), 'original_accepted': False})
all_rows = accepted_raw + new_rows
reviews = accepted_reviews + new_reviews
assert len({r['id'] for r in all_rows}) == len(all_rows)
assert len({r['id'] for r in reviews}) == len(reviews)
assert json.loads((old / 'generation_manifest.json').read_text())['inputs']['source_registry_sha256'] == json.loads((new / 'generation_manifest.json').read_text())['inputs']['source_registry_sha256']
resolver.assert_unchanged()
binding = {'original_accepted_count': len(accepted_raw), 'new_candidate_count': len(new_rows),
           'inputs': {str(folder / name): sha(folder / name) for folder, names in [
               (old, ['clean.jsonl', 'generated.jsonl', 'grounding_reviews.jsonl', 'generation_manifest.json']),
               (new, ['generated.jsonl', 'grounding_reviews.jsonl', 'generation_manifest.json'])] for name in names},
           'original_records_rewritten': False, 'same_original_source_registry': True,
           'next_step': 'Full verify, decontam, raw-question dedup, length and balance gates required', 'lineage': lineage}
stamp = out / 'combination_binding.json'
if stamp.exists():
    assert json.loads(stamp.read_text()) == binding, 'Combination inputs changed'
else:
    stamp.write_text(json.dumps(binding, indent=2))
for filename, values in [('generated.jsonl', all_rows), ('grounding_reviews.jsonl', reviews)]:
    destination = out / filename
    data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values)
    if destination.exists():
        assert destination.read_text() == data, 'Existing combined input changed'
    else:
        destination.write_text(data)
print('Combined', len(accepted_raw), 'original accepted rows with', len(new_rows), 'new candidates; still requires verification')
