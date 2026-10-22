"""Upgrade an existing installation before any automatic round uses the old worker layout."""
import json
from pathlib import Path
import subprocess
import sys

from evidence import EvidenceError, read_json
from supervisor import atomic, now


def migrate(config_path):
    config_path = Path(config_path)
    config = read_json(config_path)
    code, runtime = Path(config['code']), config_path.parent / 'runtime'
    if config.get('pool', {}).get('enabled'):
        return config
    for directory in (code / 'runs').glob('auto_rft_*'):
        if any(directory.iterdir()):
            raise EvidenceError('Existing automatic round must finish before enabling a different generation layout: ' + str(directory))
    script = (code / 'research/kd/onpolicy_gen.sbatch').read_text()
    needle = 'research/bestofn/candidates.py'
    if script.count(needle) != 2:  # One documentation reference and one invocation in the source snapshot.
        if script.count(needle) != 1:
            raise EvidenceError('Unexpected generation script layout')
    script = script.replace(needle, str(runtime / 'sample_units.py'))
    script = script.replace('# (keep --nodes=2: the row split depends on the worker count). Output: OUT/cands_<worker>.jsonl. Resumable.',
                            '# Fixed logical units allow safe resumption with a different worker count.')
    temporary = runtime / 'onpolicy_gen.sbatch.next'
    temporary.write_text(script)
    subprocess.run(['bash', '-n', str(temporary)], check=True)
    backup = config_path.with_name('config.before-pool.json')
    if not backup.exists():
        atomic(backup, config)
    temporary.replace(runtime / 'onpolicy_gen.sbatch')
    config.update(pool={'enabled': True, 'foreground_elastic_nodes': 4, 'lookahead_rounds': 6},
                  reports={'enabled': True, 'interval_seconds': 1800}, poll_seconds=10, pool_enabled_at=now())
    atomic(config_path, config)
    return config


if __name__ == '__main__':
    config = migrate(sys.argv[1])
    print(json.dumps({'pool': config['pool'], 'reports': config['reports']}))
