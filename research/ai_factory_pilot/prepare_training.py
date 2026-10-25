"""Bind every seed to exactly the same successful, reviewed data release."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from distillkit.config import load_config
from distillkit.train import admitted_training_rows

cfg = load_config(sys.argv[1])
seed = int(sys.argv[2])
root = cfg.run_dir
report = json.loads((root / 'training_gate.json').read_text())
assert report['status'] == 'passed_for_pilot_only', 'Data gate did not pass'
source = root / 'training/verified.jsonl'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == report['sha256'], 'Reviewed data changed after certification'
rows = [json.loads(line) for line in source.read_text().splitlines()]
assert len(rows) == report['selected_rows'] and len({r['id'] for r in rows}) == len(rows)
dest = root / f'seed_{seed}'
dest.mkdir(exist_ok=True)
target = dest / 'verified.jsonl'
if target.exists():
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, 'Seed already has different inputs'
else:
    shutil.copyfile(source, target)
cfg.run_dir = dest
cfg.train.seed = seed
assert len(admitted_training_rows(cfg)) == len(rows)
binding = {'seed': seed, 'data_sha256': digest, 'selected_rows': len(rows),
           'config': cfg.model_dump(mode='json'),
           'validation': 'Whole source pages reserved before generation; no seed-dependent split.'}
stamp = dest / 'training_binding.json'
if stamp.exists():
    assert json.loads(stamp.read_text()) == binding, 'Training configuration changed'
else:
    stamp.write_text(json.dumps(binding, indent=2))
print('TRAINING_INPUTS_VERIFIED', seed, len(rows), digest, flush=True)
