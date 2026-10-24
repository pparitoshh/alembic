"""Round 23 teacher scores: per row, the teacher scores that belong to the answer the row trains on.

usage: merge_kd.py DATA.jsonl QNEXT_KD_DIR Q235_KD_DIR OUT_DIR CANDS.jsonl
Rows whose answer is Qwen3-235B's (text match with CANDS) take round 21's Qwen3-235B scores of that answer;
all other rows keep the verified answer and take round 11's Qwen3-Next scores of it. Both teachers share Qwen3-4B's
vocabulary. Rows without scores for their answer are left out of the KD set (train_kd skips them).
"""
import glob
import json
import os
import sys

import numpy as np

data, qnext, q235, out, cands = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
# A row takes Qwen3-235B scores only if its answer IS Qwen3-235B's answer (text match with the candidate file);
# a 'bestofn' field is not enough: many verified rows carry one from an earlier best-of-n round (Oct 22 bug).
new = {}
for line in open(cands):
    r = json.loads(line)
    new[r['id']] = r['candidates'][0]
src = {}
for line in open(data):
    r = json.loads(line)
    src[r['id']] = 'q235' if new.get(r['id']) == r['messages'][-1]['content'] else 'qnext'
n = {'qnext': 0, 'q235': 0}
for tag, d in (('qnext', qnext), ('q235', q235)):
    for k, f in enumerate(sorted(glob.glob(os.path.join(d, 'kd_*.npz')))):
        z = np.load(f)
        ids, lps, offsets, row_ids = z['ids'], z['lps'], z['offsets'], z['row_ids']
        keep = [i for i, rid in enumerate(row_ids) if src.get(str(rid)) == tag]
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
        n[tag] += len(keep)
print(json.dumps({'rows_by_teacher': n, 'rows': len(src)}))
