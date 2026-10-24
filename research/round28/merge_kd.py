"""Round 28 teacher scores: one KD dir whose entries are Qwen3-235B top-20 scores of exactly each row's own answer.

usage: merge_kd.py DATA.jsonl NEW_KD_DIR REPLAY_KD_DIR OUT_DIR
New rows ('round28_part' == 'new') take the scores just computed for them (score235.sbatch on new_rows.jsonl);
replay rows take round 21's scores (kd235/teacher_top20, computed on kd235/data/verified.jsonl, the file the replay
rows are copied from). A row is kept only if its answer re-tokenises (answer + '<|im_end|>', as score.py renders it)
to exactly as many positions as the stored scores; anything else is left out (train_kd skips rows without scores).
Like research/round23/merge_kd.py. Writes OUT_DIR/kd_new_*.npz, kd_rep_*.npz and, last, OUT_DIR/DONE.json.
"""
import glob
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

data, new_kd, rep_kd, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507')
part, length = {}, {}
for line in open(data):
    r = json.loads(line)
    part[r['id']] = r['round28_part']
    length[r['id']] = len(tok(r['messages'][-1]['content'] + '<|im_end|>', add_special_tokens=False)['input_ids'])
n = {'new': 0, 'replay': 0, 'length_mismatch': 0}
for tag, d, want in (('new', new_kd, 'new'), ('rep', rep_kd, 'replay')):
    for k, f in enumerate(sorted(glob.glob(os.path.join(d, 'kd_*.npz')))):
        z = np.load(f)
        ids, lps, offsets, row_ids = z['ids'], z['lps'], z['offsets'], z['row_ids']
        keep = []
        for i, rid in enumerate(row_ids):
            rid = str(rid)
            if part.get(rid) != want:
                continue
            if offsets[i + 1] - offsets[i] != length[rid]:
                n['length_mismatch'] += 1
                continue
            keep.append(i)
        if not keep:
            continue
        new_off, parts_i, parts_l = [0], [], []
        for i in keep:
            parts_i.append(ids[offsets[i]:offsets[i + 1]]); parts_l.append(lps[offsets[i]:offsets[i + 1]])
            new_off.append(new_off[-1] + offsets[i + 1] - offsets[i])
        tmp = os.path.join(out, f'kd_{tag}_{k:04d}.tmp.npz')
        np.savez(tmp, ids=np.concatenate(parts_i), lps=np.concatenate(parts_l), offsets=np.array(new_off),
                 row_ids=row_ids[keep])
        os.replace(tmp, os.path.join(out, f'kd_{tag}_{k:04d}.npz'))
        n[want] += len(keep)
want_n = {p: sum(1 for v in part.values() if v == p) for p in ('new', 'replay')}
res = {'rows_scored': n, 'rows': want_n}
with open(os.path.join(out, 'DONE.json.tmp'), 'w') as f:
    json.dump(res, f)
# at least 97% of each part must have scores (score.py skips rows over 4096 tokens)
ok = all(n[p] >= want_n[p] * 97 // 100 for p in ('new', 'replay'))
if ok:
    os.replace(os.path.join(out, 'DONE.json.tmp'), os.path.join(out, 'DONE.json'))
print(json.dumps(res))
sys.exit(0 if ok else 1)
