"""Uniform weight average ("model soup") of student checkpoints that share one ancestor (same architecture and
tokenizer); fp32 accumulation, saved in the first model's dtype with its config and tokenizer files.

usage: soup.py OUT_DIR MODEL_DIR [MODEL_DIR ...]
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

out, models = Path(sys.argv[1]), [Path(m) for m in sys.argv[2:]]
out.mkdir(parents=True, exist_ok=True)
acc, dtypes = None, {}
for m in models:
    sd = load_file(str(m / 'model.safetensors'))
    if acc is None:
        acc = {k: v.float().clone() for k, v in sd.items()}
        dtypes = {k: v.dtype for k, v in sd.items()}
    else:
        assert sd.keys() == acc.keys(), f'{m}: different parameter names'
        for k, v in sd.items():
            acc[k] += v.float()
    print('added', m, flush=True)
save_file({k: (v / len(models)).to(dtypes[k]).contiguous() for k, v in acc.items()}, str(out / 'model.safetensors'),
          metadata={'format': 'pt'})
for f in models[0].iterdir():
    if f.name not in ('model.safetensors', 'KD_TRAIN.json'):
        shutil.copy2(f, out / f.name)
(out / 'SOUP.json').write_text(json.dumps({'models': [str(m) for m in models]}, indent=1))
print('soup saved', out)
