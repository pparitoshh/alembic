phase runtime_preflight
"$VENV/bin/python" - <<'PY'
import hashlib, json, os
from pathlib import Path
repo = Path(os.environ['REPO'])
manifest = repo / 'runs/ai_factory_pilot_20261013/code_snapshot.json'
if manifest.exists():
    for name, expected in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, 'Experiment code changed: ' + name
import torch
import importlib.metadata as m
assert torch.cuda.is_available(), 'CUDA is unavailable; never fall back to CPU training'
torch.empty(1, device='cuda'); torch.cuda.synchronize()
evidence = {'gpu': torch.cuda.get_device_name(), 'packages': {p:m.version(p) for p in ['torch','transformers','trl','peft']}}
(Path(os.environ['RUN_DIR']) / ('runtime-' + os.environ.get('SLURM_JOB_ID', 'local') + '.json')).write_text(json.dumps(evidence, indent=2))
print(evidence)
PY
