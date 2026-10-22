"""Repair only the scheduler wait block in hash-verified prepared controllers."""
from pathlib import Path
import subprocess
import sys

from evidence import EvidenceError, read_json, sha256
from prepare import write_identical
from supervisor import atomic, lock, now
from wait_gpu import controller_block


OLD_BLOCK = '''wait_job() {
    while true; do
        if ! queue=$(squeue -h -j "$1" -o %i); then sleep 30; continue; fi
        [ -n "$queue" ] || break
        heartbeat; sleep 30
    done
    log "job $1: $(sacct -n -X -j "$1" --format=State,Elapsed | head -1 | xargs)"
}
'''


def migrate(config_path):
    config = read_json(config_path)
    directory = Path(config['root']) / 'ctl/autopilot'
    if not (directory / 'PAUSE').exists():
        raise EvidenceError('Pause new submissions before upgrading prepared controllers')
    changed = []
    with lock(directory / 'controller_upgrade.lock') as acquired:
        if not acquired:
            raise EvidenceError('Another controller upgrade is running')
        replacement = controller_block(config, directory / 'runtime')
        for plan_path in sorted((Path(config['code']) / 'runs').glob('auto_rft_*/AUTOPILOT_PLAN.json')):
            plan = read_json(plan_path)
            run = plan_path.parent.name
            if plan['run'] != run:
                raise EvidenceError('Prepared run name does not match its directory')
            script = Path(config['root']) / 'ctl' / run / 'controller.sh'
            original = script.read_text()
            digest = sha256(script)
            backup = directory / 'repairs/accounting-wait-v1' / run
            if replacement in original:
                if digest == plan['controller_sha256']:
                    continue
                # Recover an interrupted upgrade between replacing the script and updating its plan.
                if not (backup / 'controller.sh').exists() or sha256(backup / 'controller.sh') != plan['controller_sha256']:
                    raise EvidenceError('Upgraded controller does not match its recorded backup')
                expected = (backup / 'controller.sh').read_text().replace(OLD_BLOCK, replacement)
                if expected != original:
                    raise EvidenceError('Controller changed outside the scheduler wait block')
            else:
                if digest != plan['controller_sha256'] or original.count(OLD_BLOCK) != 1:
                    raise EvidenceError('Prepared controller changed or has an unrecognized wait block: ' + run)
                write_identical(backup / 'controller.sh', original)
                write_identical(backup / 'AUTOPILOT_PLAN.json', plan_path.read_text())
                temporary = script.with_name('controller.wait-repair.tmp')
                temporary.write_text(original.replace(OLD_BLOCK, replacement))
                subprocess.run(['bash', '-n', str(temporary)], check=True)
                temporary.replace(script)
            previous = plan['controller_sha256']
            plan['controller_sha256'] = sha256(script)
            atomic(plan_path, plan)
            receipt = {'at': now(), 'run': run, 'previous_sha256': previous,
                       'controller_sha256': plan['controller_sha256'],
                       'change': 'Replace live-queue wait with terminal accounting evidence; experiment inputs unchanged.'}
            atomic(backup / 'REPAIR.json', receipt)
            changed.append(receipt)
    return changed


if __name__ == '__main__':
    import json
    print(json.dumps(migrate(sys.argv[1]), indent=2))
