"""Rebuild the public Q&A dataset (VaibhavKKM/hpc-assistant-qa) from every factory release: rows whose source page is
under a permissive licence (CC-BY-4.0, CC0-1.0, MIT, Apache-2.0 (also with the LLVM exception), BSD-2/3-Clause, ISC,
the BSD-style MPICH licence), held-out evaluation families
removed by site (Aalto, Open MPI, CUDA Samples), duplicate questions removed. Same row format as the Oct 24 release.

A second configuration, `sharealike` (data/sharealike/train.jsonl), holds rows from Creative Commons share-alike
sources under their own licence: CC-BY-SA-3.0/4.0 and CC-BY-NC-SA-4.0 (non-commercial); the GNU parallel documentation
(dual GFDL-1.3 / CC-BY-SA-4.0) is used under CC-BY-SA-4.0. No-derivatives, GPL/LGPL/GFDL-only and ambiguous
multi-licence sources stay out.

usage: build_hfds.py OUT_DIR
"""
import collections
import glob
import json
import os
import re
import sys

R = '/leonardo_work/EUHPC_D30_031/runs/a08trc09/ai-factory-ccby-v1-20261017/code/runs'
S = '/leonardo_work/EUHPC_D30_031/shared_split/code/runs'
OK = {'CC-BY-4.0', 'CC0-1.0', 'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', 'Apache-2.0 OR MIT',
      'BSD-2-Clause-Patent', 'Apache-2.0 WITH LLVM-exception', 'MPICH COPYRIGHT (permissive BSD-style)'}
SA = {'CC-BY-NC-SA-4.0': 'CC-BY-NC-SA-4.0', 'CC-BY-SA-3.0-US': 'CC-BY-SA-3.0-US', 'CC-BY-SA-4.0': 'CC-BY-SA-4.0',
      'GFDL-1.3 / CC-BY-SA-4.0 (GNU parallel docs)': 'CC-BY-SA-4.0'}
HELD = ('aalto', 'open-mpi', 'openmpi', 'cuda-samples', 'cuda_samples')
out = sys.argv[1]
os.makedirs(f'{out}/data/sharealike', exist_ok=True)


def batch_key(f):
    b = f.split('/')[-3]
    m = re.search(r'_v(\d+)$', b)
    return (0 if 'ccby' in b else 1, int(m.group(1)) if m else 0)


files = sorted(glob.glob(f'{R}/ai_factory_*/release/verified.jsonl') + glob.glob(f'{S}/ai_factory_*/release/verified.jsonl'),
               key=batch_key)
st = collections.Counter()
lic, bat, site, teach, mode, sa_lic = (collections.Counter() for _ in range(6))
seen = set()
with open(f'{out}/data/train.jsonl', 'w') as fo, open(f'{out}/data/sharealike/train.jsonl', 'w') as fs:
    for f in files:
        b = f.split('/')[-3]
        for line in open(f):
            r = json.loads(line)
            st['rows'] += 1
            raw_lic = (r.get('source_licence') or '').strip()
            if raw_lic not in OK and raw_lic not in SA:
                st['not_permissive'] += 1
                continue
            where = ' '.join(str(r.get(k) or '') for k in ('doc_id', 'source_url', 'source_site', 'source_family')).lower()
            if any(h in where for h in HELD):
                st['held_out_family_removed'] += 1
                continue
            q = r.get('question') or ''
            if q in seen:
                st['duplicate'] += 1
                continue
            seen.add(q)
            msgs = r.get('messages') or []
            ans = next((m.get('content') or '' for m in reversed(msgs) if m.get('role') == 'assistant' and m.get('content')), '')
            row = {'id': r['id'], 'question': q, 'answer': ans, 'messages': msgs, 'mode': r.get('mode'),
                   'task': r.get('task'), 'source_url': r.get('source_url'), 'source_site': r.get('source_site'),
                   'source_licence': SA.get(raw_lic, raw_lic), 'teacher': r.get('teacher'), 'batch': b}
            if raw_lic in SA:
                fs.write(json.dumps(row, ensure_ascii=False) + '\n')
                st['sharealike'] += 1
                sa_lic[row['source_licence']] += 1
                continue
            fo.write(json.dumps(row, ensure_ascii=False) + '\n')
            st['kept'] += 1
            lic[row['source_licence']] += 1; bat[b] += 1; site[row['source_site']] += 1
            teach[row['teacher']] += 1; mode[row['mode']] += 1
stats = {**{k: st[k] for k in ('kept', 'sharealike', 'rows', 'not_permissive', 'duplicate', 'held_out_family_removed')},
         'sharealike_by_licence': sa_lic.most_common(),
         'batches': len(files), 'by_licence': lic.most_common(), 'by_batch': bat.most_common(), 'top_sites': site.most_common(40),
         'by_teacher': teach.most_common(), 'by_mode': mode.most_common()}
json.dump(stats, open(f'{out}/stats.json', 'w'), indent=1)
print(stats['sharealike_by_licence'])
print(json.dumps({k: stats[k] for k in ('kept', 'sharealike', 'rows', 'not_permissive', 'duplicate', 'held_out_family_removed', 'batches')}))
print(stats['by_licence']); print(stats['by_teacher']); print(stats['by_mode'])
