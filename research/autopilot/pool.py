"""Coordinate two GPU jobs under a shared eight-node budget, including ambiguous submissions."""
import re
from pathlib import Path

from evidence import EvidenceError, read_json
from supervisor import ACTIVE, SchedulerError, atomic, lock, now


class CapacityPending(SchedulerError):
    pass


def resource_request(args, code):
    """Read the trusted batch defaults; command-line node counts override them."""
    script = Path(args[-1])
    if not script.is_absolute():
        script = Path(code) / script
    match = re.search(r'^#SBATCH\s+--nodes=(\d+)\s*$', script.read_text(), re.M)
    nodes = int(match[1]) if match else None
    for arg in args[:-1]:
        if arg.startswith('--nodes='):
            nodes = int(arg.split('=', 1)[1])
    if nodes is None or not 1 <= nodes <= 8:
        raise EvidenceError('GPU request must declare between one and eight nodes')
    # These workers divide independent, stable output units using SLURM_NTASKS.
    elastic = (script.name in ('onpolicy_gen.sbatch', 'judge.sbatch', 'judge_gemma.sbatch')
               or (script.name == 'stage.sbatch' and any('STAGE=judge' in a for a in args)))
    return nodes, elastic


class Pool:
    def __init__(self, config, scheduler):
        self.config, self.scheduler = config, scheduler
        self.directory = Path(config['root']) / 'ctl/autopilot'
        self.path = self.directory / 'POOL.json'

    def live(self):
        result = []
        output = self.scheduler.command(['squeue', '--me', '-h', '-o', '%i|%100j|%T|%q|%D|%b'])
        for line in output.splitlines():
            p = [x.strip() for x in line.split('|')]
            if len(p) != 6 or not re.fullmatch(r'\d+(?:[_+].+)?', p[0]) or not p[4].isdigit():
                raise SchedulerError('Unexpected GPU queue response: ' + line[:160])
            if 'gpu' not in p[5] or p[5].endswith(':0'):
                continue
            result.append({'id': p[0], 'name': p[1], 'state': p[2], 'qos': p[3], 'nodes': int(p[4])})
        return result

    def reconcile(self, state):
        live = self.live()
        by_id = {j['id']: j for j in live}
        reserved, slots = sum(j['nodes'] for j in live), len(live)
        for key, lease in state['leases'].items():
            if lease.get('state') in ('complete', 'refused'):
                continue
            receipt_path = self.directory / 'submissions' / (key + '.json')
            receipt = read_json(receipt_path) if receipt_path.exists() else {}
            job = lease.get('job_id') or receipt.get('job_id')
            if receipt.get('state') == 'refused' and not job:
                lease['state'] = 'refused'
                continue
            if job:
                lease['job_id'] = job
                if job in by_id:
                    continue
                status, _ = self.scheduler.status(job)
                if status not in ACTIVE and status != 'UNKNOWN':
                    lease.update(state='complete', scheduler_state=status)
                    continue
            # A job absent from squeue but not proved terminal still owns its reservation.
            reserved += lease['nodes']
            slots += 1
        return live, reserved, slots

    def submit(self, key, args, *, role='foreground', environment=None, reserve=0):
        if not self.config.get('pool', {}).get('enabled'):
            return self.scheduler.submit(key, args[-1], gpu_args=args, environment=environment)
        with lock(self.directory / 'gpu_pool.lock') as acquired:
            if not acquired:
                raise CapacityPending('GPU pool is processing another request')
            if any((self.directory / name).exists() for name in ('PAUSE', 'STOP')):
                raise CapacityPending('GPU pool submissions paused by operator control')
            state = read_json(self.path) if self.path.exists() else {'leases': {}}
            live, used, slots = self.reconcile(state)
            existing = state['leases'].get(key)
            if existing and existing.get('state') != 'refused':
                # Reuse the exact allocation intent after a crash or lost scheduler response.
                nodes = existing['nodes']
            else:
                wanted, elastic = resource_request(args, self.config['code'])
                available = max(0, 8 - used)
                if role == 'background':
                    available = max(0, available - reserve)
                elif elastic:
                    wanted = min(wanted, self.config['pool'].get('foreground_elastic_nodes', 4))
                nodes = min(wanted, available) if elastic else wanted
                minimum = 1 if elastic else wanted
                if slots >= 2 or available < minimum:
                    state.update(updated_at=now(), live_jobs=live, reserved_nodes=used,
                                 waiting={'key': key, 'role': role, 'wanted': wanted, 'minimum': minimum})
                    atomic(self.path, state)
                    raise CapacityPending(f'GPU pool: {used}/8 nodes and {slots}/2 job slots reserved; waiting for {role}')
                state['leases'][key] = {'nodes': nodes, 'role': role, 'state': 'intent', 'created_at': now()}
                atomic(self.path, state)
            actual = [a for a in args[:-1] if not a.startswith('--nodes=')]
            actual += ['--nodes=' + str(nodes), args[-1]]
            job = self.scheduler.submit(key, args[-1], gpu_args=actual, environment=environment)
            state['leases'][key].update(job_id=job, state='submitted')
            state.update(updated_at=now(), live_jobs=live)
            state.pop('waiting', None)
            atomic(self.path, state)
            return job

    def summary(self):
        with lock(self.directory / 'gpu_pool.lock') as acquired:
            if not acquired:
                return {'status': 'updating'}
            state = read_json(self.path) if self.path.exists() else {'leases': {}}
            live, used, slots = self.reconcile(state)
            running = sum(j['nodes'] for j in live if j['state'] in ('RUNNING', 'COMPLETING'))
            state.update(updated_at=now(), live_jobs=live, reserved_nodes=used)
            atomic(self.path, state)
            return {'running_nodes': running, 'running_gpus': 4 * running,
                    'reserved_nodes': used, 'jobs': slots, 'limit_nodes': 8, 'live_jobs': live}
