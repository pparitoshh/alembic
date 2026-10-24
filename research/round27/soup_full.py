"""Round 27: uniform weight soup on top of the round 20 release.

usage: soup_full.py BASE_DIR OUT_DIR ADAPTER [ADAPTER ...]
OUT = BASE + mean_i(delta_i), delta_i = (alpha_i / r_i) * B_i @ A_i for every LoRA target module of adapter i.
Adapters may have different ranks (exact: each update is formed in full before averaging). All adapters must be
trained on BASE. Uniform over every given adapter (no selection on test scores).
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

base, out, srcs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(a) for a in sys.argv[3:]]
if (out / 'model.safetensors').exists():
    sys.exit(0)
# Keep only the small LoRA factors in memory and form each module's mean update when its base weight is read
# (peak memory ~ base model + adapters, fits the 30 GB CPU-partition limit).
factors = defaultdict(list)
for s in srcs:
    cfg = json.loads((s / 'adapter_config.json').read_text())
    assert not cfg.get('use_rslora') and not cfg.get('use_dora') and not cfg.get('rank_pattern') and not cfg.get('alpha_pattern'), s
    scale = cfg['lora_alpha'] / cfg['r']
    t = load_file(s / 'adapter_model.safetensors')
    for k, a in t.items():
        if '.lora_A.' in k:
            name = k.split('base_model.model.', 1)[1].replace('.lora_A.weight', '.weight')
            factors[name].append((a, t[k.replace('.lora_A.', '.lora_B.')], scale))
    print('loaded', s, flush=True)
n = len(srcs)
tmp = out.with_name(out.name + '.tmp')
tmp.mkdir(parents=True, exist_ok=True)
weights, touched = {}, 0
with safe_open(base / 'model.safetensors', 'pt') as f:
    for k in f.keys():
        w = f.get_tensor(k)
        if k in factors:
            d = sum((b.float() @ a.float()) * sc for a, b, sc in factors[k])
            w = (w.float() + d / n).to(w.dtype); touched += 1
        weights[k] = w
missing = [k for k in factors if k not in weights]
assert not missing, f'LoRA targets missing in base: {missing[:3]}'
save_file(weights, tmp / 'model.safetensors', metadata={'format': 'pt'})
for p in base.iterdir():
    if p.name != 'model.safetensors' and p.is_file():
        shutil.copy(p, tmp / p.name)
(tmp / 'SOUP.json').write_text(json.dumps({'base': str(base), 'adapters': [str(s) for s in srcs], 'modules': touched}, indent=1))
tmp.rename(out)
print(f'soup of {n} adapters, {touched} modules -> {out}')
