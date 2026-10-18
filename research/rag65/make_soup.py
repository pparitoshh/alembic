"""Seed soup: one LoRA equal to the mean of the seeds' weight updates, scaled by s.

usage: make_soup.py OUT_DIR SCALE ADAPTER [ADAPTER ...]
Seeds start from different random A matrices, so averaging A and B separately is not the mean update.
Stacking is exact: mean_i(B_i A_i) = [B_1 ... B_n] / n . [A_1; ...; A_n], a rank n*r adapter. lora_alpha
grows with r so the alpha/r scaling stays the same (plain LoRA, no rsLoRA).
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

out, scale, srcs = Path(sys.argv[1]), float(sys.argv[2]), [Path(a) for a in sys.argv[3:]]
if (out / 'adapter_model.safetensors').exists():
    sys.exit(0)
cfgs = [json.loads((s / 'adapter_config.json').read_text()) for s in srcs]
r, alpha = cfgs[0]['r'], cfgs[0]['lora_alpha']
assert all(c['r'] == r and c['lora_alpha'] == alpha and not c.get('use_rslora') and not c.get('use_dora')
           and not c.get('rank_pattern') and not c.get('alpha_pattern') for c in cfgs), 'identical plain LoRA configs expected'
tensors = [load_file(s / 'adapter_model.safetensors') for s in srcs]
assert all(t.keys() == tensors[0].keys() for t in tensors)
n = len(srcs)
soup = {}
for k in tensors[0]:
    if 'lora_A' in k:
        soup[k] = torch.cat([t[k] for t in tensors], dim=0)
    elif 'lora_B' in k:
        soup[k] = torch.cat([t[k] for t in tensors], dim=1) * (scale / n)
    else:
        raise SystemExit('unexpected tensor ' + k)
out.mkdir(parents=True)
for f in srcs[0].iterdir():
    if f.is_file() and f.name not in ('adapter_model.safetensors', 'adapter_config.json', 'SCALE.json'):
        shutil.copy(f, out / f.name)
cfg = dict(cfgs[0], r=r * n, lora_alpha=alpha * n)
(out / 'adapter_config.json').write_text(json.dumps(cfg, indent=2))
save_file(soup, out / 'adapter_model.safetensors')
(out / 'SOUP.json').write_text(json.dumps({'sources': [str(s) for s in srcs], 'scale': scale, 'rank': r * n}))
