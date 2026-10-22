"""Prepare future, verified RFT datasets alongside the foreground GPU experiment."""
import json
from pathlib import Path
import subprocess
import sys

from evidence import EvidenceError, read_json, rows, sha256
from pool import CapacityPending, Pool
from prepare import prepare
from sample_units import UNIT_SIZE, completed, unit_rows
from supervisor import ACTIVE, RETRYABLE, atomic, now


class IncompleteUnits(EvidenceError):
    """Expected work is missing; malformed or unexpected evidence is a separate error."""


def unit_progress(directory, stage):
    if stage == 'judge':
        return len(list((directory / 'judged_teacher').glob('unit_*.jsonl')))
    complete = len(list((directory / 'gen').glob('cands_unit_*.jsonl'))) * UNIT_SIZE
    return complete + sum(p.read_bytes().count(b'\n') for p in (directory / 'gen').glob('cands_unit_*.partial'))


def ready_proof(directory):
    directory = Path(directory)
    proof = read_json(directory / 'DATA_READY.json')
    if proof['rows'] < 500 or proof['threshold'] != .75:
        raise EvidenceError('Prepared dataset does not meet the fixed acceptance gates')
    for name, digest in [('data/questions.jsonl', proof['question_sha256']),
                         ('data/verified.jsonl', proof['sha256']),
                         ('SAMPLES_READY.json', proof['sample_manifest_sha256'])]:
        if sha256(directory / name) != digest:
            raise EvidenceError('Prepared dataset changed: ' + name)
    for name, digest in proof['judgments']['files'].items():
        if sha256(directory / 'judged_teacher' / name) != digest:
            raise EvidenceError('Training judgments changed: ' + name)
    return proof


def sample_proof(directory):
    counts, digests = {}, {}
    units = unit_rows(directory / 'data/questions.jsonl')
    expected_files = {f'cands_unit_{i:05d}.jsonl' for i in range(len(units))}
    actual_files = {p.name for p in (directory / 'gen').glob('cands_*.jsonl')}
    if actual_files - expected_files:
        raise EvidenceError('Unexpected training sample files')
    if actual_files != expected_files:
        raise IncompleteUnits(f'Sampling has {len(actual_files)}/{len(expected_files)} complete units')
    for number, unit in enumerate(units):
        p = directory / 'gen' / f'cands_unit_{number:05d}.jsonl'
        result = completed(p, {r['id'] for r in unit})
        counts.update({rid: len(row['candidates']) for rid, row in result.items()})
        digests[p.name] = sha256(p)
    return {'questions': len(counts), 'candidate_counts': counts, 'files': digests,
            'question_sha256': sha256(directory / 'data/questions.jsonl')}


def judgment_proof(directory, samples):
    for name, digest in samples['files'].items():
        if sha256(directory / 'gen' / name) != digest:
            raise EvidenceError('Training samples changed after generation: ' + name)
    ids = sorted(rid for rid, count in samples['candidate_counts'].items() if count)
    units = [ids[i:i + 25] for i in range(0, len(ids), 25)]
    expected_files = {f'unit_{i:05d}.jsonl' for i in range(len(units))}
    actual_files = {p.name for p in (directory / 'judged_teacher').glob('unit_*.jsonl')}
    if actual_files - expected_files:
        raise EvidenceError('Unexpected training judgment files')
    if actual_files != expected_files:
        raise IncompleteUnits(f'Training judging has {len(actual_files)}/{len(expected_files)} complete units')
    invalid, count, digests = 0, 0, {}
    for number, unit in enumerate(units):
        path = directory / 'judged_teacher' / f'unit_{number:05d}.jsonl'
        expected = {(rid, k) for rid in unit for k in range(samples['candidate_counts'][rid])}
        seen = set()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['id'], row['k'])
            if key not in expected or key in seen:
                raise EvidenceError('Unexpected or repeated training judgment: ' + str(path))
            score = row.get('score')
            if isinstance(score, bool) or score not in (None, 0, .25, .5, .75, 1):
                raise EvidenceError('Invalid training preference score')
            invalid += score is None
            seen.add(key)
        if seen != expected:
            raise EvidenceError('Incomplete training judgment unit: ' + str(path))
        count += len(seen)
        digests[path.name] = sha256(path)
    return {'judgments': count, 'invalid': invalid, 'units': len(units), 'files': digests}


class Backfill:
    def __init__(self, supervisor):
        self.s = supervisor
        self.config = supervisor.config
        self.directory = supervisor.directory
        self.runtime = supervisor.runtime
        self.code = supervisor.code
        self.path = self.directory / 'BACKFILL.json'
        self.state = read_json(self.path) if self.path.exists() else {'runs': {}, 'active': None}
        self.pool = Pool(self.config, supervisor.slurm)

    def save(self):
        self.state['updated_at'] = now()
        atomic(self.path, self.state)
        self.s.state['backfill'] = {k: v for k, v in self.state.items() if k != 'runs'}
        self.s.state['gpu_pool'] = self.pool.summary()

    def reserve_foreground(self, live):
        # A running foreground allocation already defines its share. Otherwise reserve its next step.
        pool_state = read_json(self.pool.path) if self.pool.path.exists() else {'leases': {}}
        background = {x.get('job_id') for x in pool_state['leases'].values() if x.get('role') == 'background'}
        if any(j['id'] not in background for j in live):
            return 0
        phase = self.s.state['phase']
        if phase == 'history':
            return 4
        if phase in ('prepare', 'train_evaluate'):
            run = f"auto_rft_{self.s.state['round_index']:04d}"
            path = self.code / 'runs' / run
            if not (path / 'DATA_READY.json').exists():
                return 0
            return 3 if len(list((path / 'train').glob('seed*/TRAINING_DONE.json'))) < 3 else 4
        if phase == 'compare':
            return 4
        if phase in ('candidate_checks', 'export'):
            return 1
        return 0

    def tick(self):
        active = self.state.get('active')
        if active:
            record = self.state['runs'][active]
            jobs = [record['job_id']] + ([record['helper_job_id']] if record.get('helper_job_id') else [])
            statuses = {job: self.s.slurm.status(job) for job in jobs}
            status, exit_code = statuses[record['job_id']]
            record['scheduler_state'] = status
            helper = record.get('helper_job_id')
            if helper and statuses[helper][0] in ('PENDING', 'RUNNING', 'CONFIGURING'):
                self.release_surplus_helper(active, record)
            expected_cancel = lambda job: job == helper and record.get('helper_cancel_requested', False)
            if any(st == 'CANCELLED' and not expected_cancel(job) for job, (st, _) in statuses.items()):
                raise EvidenceError(f'Background job for {active} was cancelled; respecting cancellation')
            if any(st in ACTIVE or st == 'UNKNOWN' for st, _ in statuses.values()):
                if status in ACTIVE:
                    self.help_sample(active, record)
                run = self.code / 'runs' / active
                pattern = 'gen/cands_unit_*.jsonl' if record['stage'] == 'sample' else 'judged_teacher/unit_*.jsonl'
                names = record['job_id'] + (' + ' + record['helper_job_id'] if record.get('helper_job_id') else '')
                self.state['detail'] = f"{active}: {record['stage']}, jobs {names}, {len(list(run.glob(pattern)))} completed units"
                self.save()
                return
            for job, (st, code) in statuses.items():
                if st == 'CANCELLED' and expected_cancel(job):
                    continue
                if st not in RETRYABLE and not (st == 'COMPLETED' and code == '0:0'):
                    record['failures'] = record.get('failures', 0) + 1
            progress = unit_progress(self.code / 'runs' / active, record['stage'])
            if status in ('COMPLETED', 'TIMEOUT'):
                record['stalls'] = record.get('stalls', 0) + 1 if progress <= record.get('progress_before', -1) else 0
            record['previous_job_id'] = record.pop('job_id')
            if record.get('helper_job_id'):
                record['previous_helper_job_id'] = record.pop('helper_job_id')
                record.pop('helper_cancel_requested', None)
            self.state['active'] = None
            self.save()
            if record.get('failures', 0) >= 3:
                raise EvidenceError(f'{active} background preparation failed three times; inspect its logs')
            if record.get('stalls', 0) >= 3:
                raise EvidenceError(f'{active} background preparation made no progress in three allocations; inspect its logs')

        start = self.s.state['round_index']
        selected = None
        for index in range(start, start + self.config['pool'].get('lookahead_rounds', 6)):
            run = f'auto_rft_{index:04d}'
            directory = self.code / 'runs' / run
            if not (directory / 'DATA_READY.json').exists() and not (directory / 'DATA_REJECTED.json').exists():
                selected = index, run, directory
                break
        if not selected:
            self.state['detail'] = 'All configured future datasets are ready; waiting for foreground progress.'
            self.save()
            return
        index, run, directory = selected
        if not (directory / 'AUTOPILOT_PLAN.json').exists():
            self.state['detail'] = 'Preparing protected training questions for ' + run
            self.save()
            prepare(self.config, index)
        plan = read_json(directory / 'AUTOPILOT_PLAN.json')
        if plan.get('generation_layout') != 'fixed-units-v1':
            raise EvidenceError(f'{run} uses the previous fixed-worker layout; finish it before enabling elastic backfill')
        if sha256(directory / 'data/questions.jsonl') != plan['question_sha256']:
            raise EvidenceError('Prepared training questions changed')
        record = self.state['runs'].setdefault(run, {'index': index, 'attempt': 0, 'failures': 0})
        sample_path = directory / 'SAMPLES_READY.json'
        if not sample_path.exists():
            try:
                samples = sample_proof(directory)
            except IncompleteUnits:
                stage = 'sample'
            else:
                atomic(sample_path, samples)
                stage = 'judge'
        else:
            samples = read_json(sample_path)
            stage = 'judge'
        if stage == 'judge':
            try:
                judged = judgment_proof(directory, samples)
            except IncompleteUnits:
                pass
            else:
                self.build(directory, plan, samples, judged)
                record.update(stage='ready' if (directory / 'DATA_READY.json').exists() else 'rejected', completed_at=now())
                self.state['detail'] = run + ': ' + record['stage']
                self.save()
                return
        if record.get('stage') != stage:
            record['failures'] = 0
            record['stalls'] = 0
        record['stage'] = stage
        args, env = self.request(directory, plan, stage)
        try:
            reserve = self.reserve_foreground(self.pool.live())
            attempt = record['attempt'] + 1
            key = f'background-{run}-{stage}-{attempt}'
            job = self.pool.submit(key, args, role='background', environment=env, reserve=reserve)
        except CapacityPending as exc:
            self.state['detail'] = str(exc)
        else:
            record.update(job_id=job, attempt=attempt, submitted_at=now(), progress_before=unit_progress(directory, stage))
            self.state.update(active=run, detail=f'{run}: {stage}, job {job}')
        self.save()

    def request(self, directory, plan, stage):
        env = {'REPO': str(self.code), 'RUN': plan['run'], 'MODEL': str(self.code / self.config['baseline_model']),
               'DATA': str(directory / 'data/questions.jsonl'), 'K': '4',
               'AUTOPILOT_SAMPLE_SEED': str(plan['sample_seed']),
               'AUTOPILOT_TRAINING_NOTE': plan['training_only_note']}
        if stage == 'sample':
            env['OUT'] = str(directory / 'gen')
            script = self.runtime / 'onpolicy_gen.sbatch'
        else:
            env.update(CANDS=str(directory / 'gen'), OUT=str(directory / 'judged_teacher'))
            script = self.code / 'research/rft/judge.sbatch'
        return ['--nodes=8', '--export=ALL', str(script)], env

    def help_sample(self, run, record):
        # When training is blocked on this very dataset, use the second slot to help it.
        # Unit locks make two samplers safe; judges must never use this path.
        if record['stage'] != 'sample' or record.get('helper_job_id'):
            return
        if self.s.state['phase'] not in ('prepare', 'train_evaluate') or record['index'] != self.s.state['round_index']:
            return
        directory = self.code / 'runs' / run
        if (directory / 'DATA_READY.json').exists():
            return
        plan = read_json(directory / 'AUTOPILOT_PLAN.json')
        if sha256(directory / 'data/questions.jsonl') != plan['question_sha256']:
            raise EvidenceError('Prepared training questions changed')
        if len(list((directory / 'gen').glob('cands_unit_*.jsonl'))) >= (plan['questions'] + UNIT_SIZE - 1) // UNIT_SIZE:
            return
        args, env = self.request(directory, plan, 'sample')
        try:
            record['helper_job_id'] = self.pool.submit(f"background-{run}-sample-helper-{record['attempt']}",
                                                      args, role='background', environment=env)
        except CapacityPending:
            pass

    def release_surplus_helper(self, run, record):
        """Release only our optional sampler, and only after all samples are proved complete."""
        if record['stage'] != 'sample':
            return
        directory = self.code / 'runs' / run
        plan = read_json(directory / 'AUTOPILOT_PLAN.json')
        if len(list((directory / 'gen').glob('cands_unit_*.jsonl'))) < (plan['questions'] + UNIT_SIZE - 1) // UNIT_SIZE:
            return
        samples = sample_proof(directory)
        if samples['question_sha256'] != plan['question_sha256']:
            raise EvidenceError('Prepared training questions changed')
        # Persist the deliberate cancellation before contacting Slurm, so recovery distinguishes it
        # from a user's cancellation. No foreground allocation or unrelated job is touched.
        record['helper_cancel_requested'] = True
        self.save()
        self.s.slurm.command(['scancel', record['helper_job_id']])

    def build(self, directory, plan, samples, judged):
        if sha256(directory / 'data/questions.jsonl') != samples['question_sha256']:
            raise EvidenceError('Training questions changed after generation')
        out = directory / 'data/verified.build.jsonl'
        log = Path(self.config['root']) / 'ctl' / plan['run'] / 'build.out'
        with log.open('w') as f:
            subprocess.run([self.config['python'], str(self.code / 'research/rft/build.py'),
                str(directory / 'data/questions.jsonl'), str(directory / 'gen'), str(directory / 'judged_teacher'),
                str(out), '.75'], cwd=self.code, stdout=f, stderr=subprocess.STDOUT, check=True)
        accepted = rows(out)
        for rid, row in accepted.items():
            if rid not in samples['candidate_counts'] or row.get('rft', {}).get('score_vs_teacher', 0) < .75:
                raise EvidenceError('Training builder violated the acceptance threshold')
        proof = {'at': now(), 'run': plan['run'], 'rows': len(accepted), 'threshold': .75,
                 'minimum_rows': 500, 'question_sha256': samples['question_sha256'], 'sha256': sha256(out),
                 'sample_manifest_sha256': sha256(directory / 'SAMPLES_READY.json'), 'judgments': judged}
        if len(accepted) < 500:
            atomic(directory / 'DATA_REJECTED.json', {**proof, 'reason': 'Insufficient accepted training rows; threshold unchanged.'})
        else:
            out.replace(directory / 'data/verified.jsonl')
            atomic(directory / 'DATA_READY.json', proof)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] != '--check-ready':
        raise SystemExit('Usage: backfill.py --check-ready RUN_DIRECTORY')
    print(json.dumps(ready_proof(sys.argv[2])))
