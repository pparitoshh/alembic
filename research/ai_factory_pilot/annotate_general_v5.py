"""After merge_factory_v1.py for a general-HPC batch: add per-row topic and licence class, rebuild the permissive file.

usage: [BATCH=ai_factory_all_v5] annotate_general_v5.py
merge_factory_v1.py copies source_licence/source_url/source_site and writes verified_permissive.jsonl without
'unspecified' and NC-SA rows. The general archive also has proprietary, copyright-notice and compound licence strings
(licence_class 'other_needs_review' in licences.json): those stay out of the permissive file too. Rows also get
source_topic, source_id and source_licence_class. REPORT.json gets the new sha256 and per-topic/class counts.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from distillkit.io import read_jsonl, write_jsonl

BATCH = os.environ.get('BATCH', 'ai_factory_all_v5')
root = Path('runs') / BATCH
out = root / os.environ.get('RELEASE_DIR', 'release')
licences = json.loads((root / 'licences.json').read_text())
rows = read_jsonl(out / 'verified.jsonl')
for r in rows:
    meta = licences[r['doc_id']]
    r['source_topic'], r['source_id'] = meta['topic'], meta['source_id']
    r['source_licence_class'] = meta['licence_class']
write_jsonl(out / 'verified.jsonl', rows)
permissive = [r for r in rows if r['source_licence_class'] in ('permissive', 'copyleft')]
write_jsonl(out / 'verified_permissive.jsonl', permissive)
report = json.loads((out / 'REPORT.json').read_text())
report.update({
    'licence_scope': 'general HPC sources; per-row source_licence/source_licence_class/source_topic; '
                     'verified_permissive.jsonl keeps licence_class permissive and copyleft only',
    'by_topic': dict(Counter(r['source_topic'] for r in rows).most_common()),
    'by_licence_class': dict(Counter(r['source_licence_class'] for r in rows).most_common()),
    'permissive_rows': len(permissive),
    'sha256': hashlib.sha256((out / 'verified.jsonl').read_bytes()).hexdigest(),
    'annotated_by': 'annotate_general_v5.py', 'annotated_at': datetime.now(timezone.utc).isoformat()})
(out / 'REPORT.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ('released_rows', 'by_topic', 'by_licence_class', 'permissive_rows', 'sha256')}, indent=2))
