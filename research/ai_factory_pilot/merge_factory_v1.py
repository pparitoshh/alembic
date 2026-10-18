"""Merge the reviewed all-factory shards into one fail-closed, licence-tagged release. CPU only.

usage: [BATCH=ai_factory_all_v2] merge_factory_v1.py SHARDS BANK.jsonl [BANK.jsonl ...]
Same gates as the CC-BY merges (source binding, positive grounding review, eval overlap, duplicates,
existing-bank repeats, token length). Near-duplicate search uses a shingle index, since all-pairs Jaccard
over ~100K questions is too slow. Every released row gets source_licence/source_url from licences.json.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
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
BATCH = os.environ.get('BATCH', 'ai_factory_all_v1')
root = Path('runs') / BATCH
out = root / os.environ.get('RELEASE_DIR', 'release')  # a dry run writes elsewhere
out.mkdir(parents=True, exist_ok=True)
licences = json.loads((root / 'licences.json').read_text())


class NearDup:
    """Jaccard >= threshold lookup over shingle sets via an inverted index."""

    def __init__(self, threshold=0.8):
        self.threshold, self.sets, self.index = threshold, [], defaultdict(list)

    def add(self, sh):
        for s in sh:
            self.index[s].append(len(self.sets))
        self.sets.append(sh)

    def hit(self, sh):
        shared = Counter(i for s in sh for i in self.index.get(s, ()))
        # Jaccard >= t needs at least t * |sh| shared shingles.
        need = self.threshold * len(sh)
        return any(n >= need and _jaccard(sh, self.sets[i]) >= self.threshold for i, n in shared.items())


def raw_question(row):
    return row.get('question_generation', {}).get('raw_question', row['question'])


held, bank, seen = NearDup(), NearDup(), NearDup()
for _, _, h in heldout_questions(EVAL_ROOT):
    held.add(h)
for b in banks:
    for r in read_jsonl(Path(b)):
        q = raw_question(r) if 'question' in r else next(m['content'] for m in r['messages'] if m['role'] == 'user')
        bank.add(_shingles(q))
accepted, rejected, per_shard = [], [], {}
for k in range(shards):
    cfg = load_config('configs/%s_shard_%03d.json' % (BATCH, k))
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
        if held.hit(sh):
            reason = 'raw_question_eval_overlap'
        elif seen.hit(sh):
            reason = 'raw_question_duplicate'
        elif bank.hit(sh):
            reason = 'raw_question_in_existing_training_bank'
        elif not final_answer(row).strip():
            reason = 'missing_final_answer'
        if reason:
            rejected.append({'id': row['id'], 'reason': reason})
        else:
            meta = licences[row['doc_id']]
            row['source_licence'], row['source_url'], row['source_site'] = meta['licence'], meta['url'], meta['site']
            accepted.append(row)
            seen.add(sh)
    resolver.assert_unchanged()

from transformers import AutoTokenizer
cfg = load_config('configs/%s_shard_000.json' % BATCH)
tok = AutoTokenizer.from_pretrained(cfg.student.model, local_files_only=True)
lengths = token_lengths(tok, [training_example(r, cfg.student.system_prompt) for r in accepted])
release = []
for row, length in zip(accepted, lengths, strict=True):
    if length > cfg.train.max_length:
        rejected.append({'id': row['id'], 'reason': 'would_truncate', 'tokens': length})
    else:
        release.append(row)
write_jsonl(out / 'verified.jsonl', release)
write_jsonl(out / 'verified_permissive.jsonl',
            [r for r in release if r['source_licence'] not in ('unspecified', 'CC-BY-NC-SA-4.0')])
modes = Counter(r.get('mode') for r in release)
report = {
    'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'automated_gates_passed_not_human_reviewed',
    'licence_scope': 'all factory sources; per-row source_licence; verified_permissive.jsonl excludes NC-SA and unspecified',
    'clean_rows_per_shard': per_shard, 'released_rows': len(release), 'modes': dict(modes),
    'tool_fraction': round(sum(v for m, v in modes.items() if m != 'prose') / max(1, len(release)), 3),
    'by_licence': dict(Counter(r['source_licence'] for r in release)),
    'by_site': dict(Counter(r['source_site'] for r in release).most_common(40)),
    'source_documents': len({r['doc_id'] for r in release}),
    'rejections': dict(Counter(r['reason'] for r in rejected)), 'bank_rows': len(bank.sets),
    'median_answer_chars': sorted(len(final_answer(r)) for r in release)[len(release) // 2] if release else 0,
    'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest()}
(out / 'REJECTED.json').write_text(json.dumps(rejected, indent=1))
(out / 'REPORT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
