"""Scaled adapters W = base + s * (W_student - base): multiply every lora_B by s (LoRA is linear in B)."""
import json
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

src, scale = Path(sys.argv[1]), float(sys.argv[2])
dst = src.parent / ('adapter_x%02d' % round(scale * 10))
if (dst / 'adapter_model.safetensors').exists():
    sys.exit(0)
dst.mkdir()
for f in src.iterdir():
    if f.name != 'adapter_model.safetensors' and f.is_file():
        shutil.copy(f, dst / f.name)
tensors = load_file(src / 'adapter_model.safetensors')
assert any('lora_B' in k for k in tensors) and not any('magnitude' in k for k in tensors), 'plain LoRA expected'
save_file({k: (v * scale if 'lora_B' in k else v) for k, v in tensors.items()}, dst / 'adapter_model.safetensors')
(dst / 'SCALE.json').write_text(json.dumps({'source': str(src), 'scale': scale}))
