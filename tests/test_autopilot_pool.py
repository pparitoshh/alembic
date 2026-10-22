"""Exercise real scheduler-boundary failures and evidence, without model/GPU dependencies."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research/autopilot'))
import backfill as bf
import evidence as ev
import pool
import sample_units as samples
import supervisor as sup
import reports
import enable_pool


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


class Scheduler:
    def __init__(self):
        self.jobs = []
        self.submissions = {}
        self.states = {}
        self.offline = False

    def command(self, args):
        if self.offline:
            raise sup.SchedulerError('offline')
        return '\n'.join(f"{j['id']}|fixture|RUNNING|boost_qos_dbg|{j['nodes']}|gres/gpu:4" for j in self.jobs)

    def status(self, job):
        return self.states.get(job, ('RUNNING', '0:0'))

    def submit(self, key, script, *, gpu_args=None, environment=None):
        if key in self.submissions:
            return self.submissions[key]['id']
        nodes = int(next(a.split('=')[1] for a in gpu_args if a.startswith('--nodes=')))
        job = {'id': str(100 + len(self.submissions)), 'nodes': nodes, 'environment': environment}
        self.submissions[key] = job
        self.jobs.append(job)
        return job['id']


@pytest.fixture
def shared(tmp_path):
    config = {'root': str(tmp_path), 'code': str(tmp_path / 'code'), 'pool': {'enabled': True}}
    p = pool.Pool(config, Scheduler())
    scripts = tmp_path / 'code'
    scripts.mkdir()
    for name, nodes in [('train.sbatch', 3), ('stage.sbatch', 2), ('judge.sbatch', 8), ('onpolicy_gen.sbatch', 8)]:
        (scripts / name).write_text('#!/bin/bash\n#SBATCH --nodes=' + str(nodes) + '\n')
    return p


@pytest.mark.parametrize('fixed,expected', [('train.sbatch', 3), ('stage.sbatch', 2), ('judge.sbatch', 4)])
def test_two_queues_share_eight_nodes(shared, fixed, expected):
    shared.submit('main', [fixed])
    shared.submit('background', ['onpolicy_gen.sbatch'], role='background')
    assert [j['nodes'] for j in shared.scheduler.jobs] == [expected, 8 - expected]
    with pytest.raises(pool.CapacityPending):
        shared.submit('third', ['judge.sbatch'])
    assert shared.summary()['running_gpus'] == 32


def test_fixed_training_job_cannot_be_shrunk(shared):
    shared.scheduler.jobs.append({'id': '999', 'nodes': 6})
    with pytest.raises(pool.CapacityPending):
        shared.submit('train', ['train.sbatch'])
    assert not shared.scheduler.submissions


def test_background_leaves_room_for_foreground(shared):
    shared.submit('background', ['judge.sbatch'], role='background', reserve=4)
    shared.submit('main', ['judge.sbatch'])
    assert [j['nodes'] for j in shared.scheduler.jobs] == [4, 4]


def test_foreground_judge_fills_last_slot_after_background_allocation_shrinks(shared):
    shared.submit('background', ['judge.sbatch'], role='background', reserve=5)
    shared.submit('main', ['judge.sbatch'])
    assert [j['nodes'] for j in shared.scheduler.jobs] == [3, 5]
    assert shared.summary()['running_gpus'] == 32


def test_unknown_job_absent_from_queue_keeps_its_reservation(shared):
    sup.atomic(shared.path, {'leases': {'lost': {'state': 'submitted', 'nodes': 8, 'job_id': '77'}}})
    shared.scheduler.states['77'] = ('UNKNOWN', None)
    with pytest.raises(pool.CapacityPending):
        shared.submit('new', ['judge.sbatch'])
    assert not shared.scheduler.submissions
    shared.scheduler.states['77'] = ('COMPLETED', '0:0')
    shared.submit('new', ['judge.sbatch'])
    assert shared.scheduler.submissions['new']['nodes'] == 4


def test_ambiguous_intent_reserves_capacity_and_retry_keeps_exact_size(shared):
    sup.atomic(shared.path, {'leases': {'lost': {'state': 'intent', 'nodes': 5, 'role': 'background'}}})
    sup.atomic(shared.directory / 'submissions/lost.json', {'state': 'unknown'})
    shared.submit('main', ['train.sbatch'])
    with pytest.raises(pool.CapacityPending):
        shared.submit('third', ['judge.sbatch'])
    # The scheduler's receipt handler performs recovery. Pool must pass the original size.
    shared.submit('lost', ['onpolicy_gen.sbatch'], role='background')
    assert shared.scheduler.submissions['lost']['nodes'] == 5
    assert sum(j['nodes'] for j in shared.scheduler.jobs) == 8


def test_scheduler_failure_or_operator_pause_never_submits(shared):
    shared.scheduler.offline = True
    with pytest.raises(sup.SchedulerError, match='offline'):
        shared.submit('new', ['judge.sbatch'])
    shared.scheduler.offline = False
    (shared.directory / 'PAUSE').touch()
    with pytest.raises(pool.CapacityPending, match='paused'):
        shared.submit('new', ['judge.sbatch'])
    assert not shared.scheduler.submissions


def test_worker_resize_preserves_units_and_partial_recovery(tmp_path):
    data = tmp_path / 'questions.jsonl'
    records = [{'id': str(i)} for i in range(130)]
    jsonl(data, records)
    units = samples.unit_rows(data)
    for workers in [4, 12, 32]:
        partition = [[n for n in range(len(units)) if n % workers == w] for w in range(workers)]
        assert sorted(n for work in partition for n in work) == [0, 1, 2]
    p = tmp_path / 'unit.partial'
    jsonl(p, [{'id': '0', 'candidates': []}])
    with p.open('a') as f:
        f.write('{"id":')
    assert samples.recover(p, {'0', '1'}) == {'0': {'id': '0', 'candidates': []}}
    assert p.read_text().endswith('\n')
    jsonl(p, [{'id': '0', 'candidates': []}] * 2)
    with pytest.raises(ev.EvidenceError, match='repeated'):
        samples.recover(p, {'0', '1'})


def test_complete_units_preserve_empty_samples_and_reject_wrong_ids(tmp_path):
    p = tmp_path / 'unit.jsonl'
    jsonl(p, [{'id': 'a', 'candidates': []}])
    assert samples.completed(p, {'a'})['a']['candidates'] == []
    with pytest.raises(ev.EvidenceError, match='exactly'):
        samples.completed(p, {'a', 'b'})


def test_backfill_proofs_bind_ids_candidates_and_invalid_judgments(tmp_path):
    jsonl(tmp_path / 'data/questions.jsonl', [{'id': 'a'}, {'id': 'b'}])
    with pytest.raises(bf.IncompleteUnits):
        bf.sample_proof(tmp_path)
    gen = tmp_path / 'gen/cands_unit_00000.jsonl'
    jsonl(gen, [{'id': 'a', 'candidates': ['answer']}, {'id': 'b', 'candidates': []}])
    proof = bf.sample_proof(tmp_path)
    judgments = tmp_path / 'judged_teacher/unit_00000.jsonl'
    jsonl(judgments, [{'id': 'a', 'k': 0, 'score': None}])
    assert bf.judgment_proof(tmp_path, proof)['invalid'] == 1
    jsonl(judgments, [{'id': 'a', 'k': 0, 'score': .75}] * 2)
    with pytest.raises(ev.EvidenceError, match='repeated'):
        bf.judgment_proof(tmp_path, proof)
    jsonl(gen, [{'id': 'a', 'candidates': ['changed']}, {'id': 'b', 'candidates': []}])
    with pytest.raises(ev.EvidenceError, match='changed'):
        bf.judgment_proof(tmp_path, proof)


def test_unexpected_sampling_files_are_corruption_not_missing_work(tmp_path):
    jsonl(tmp_path / 'data/questions.jsonl', [{'id': 'a'}])
    jsonl(tmp_path / 'gen/cands_old_worker.jsonl', [{'id': 'a', 'candidates': []}])
    with pytest.raises(ev.EvidenceError, match='Unexpected') as err:
        bf.sample_proof(tmp_path)
    assert not isinstance(err.value, bf.IncompleteUnits)


def test_prepared_data_changed_after_acceptance_is_rejected(tmp_path):
    jsonl(tmp_path / 'data/questions.jsonl', [{'id': 'a'}])
    jsonl(tmp_path / 'data/verified.jsonl', [{'id': 'a'}])
    sup.atomic(tmp_path / 'SAMPLES_READY.json', {})
    sup.atomic(tmp_path / 'DATA_READY.json', {'rows': 500, 'threshold': .75,
        'question_sha256': ev.sha256(tmp_path / 'data/questions.jsonl'),
        'sha256': ev.sha256(tmp_path / 'data/verified.jsonl'),
        'sample_manifest_sha256': ev.sha256(tmp_path / 'SAMPLES_READY.json'), 'judgments': {'files': {}}})
    bf.ready_proof(tmp_path)
    jsonl(tmp_path / 'data/verified.jsonl', [{'id': 'changed'}])
    with pytest.raises(ev.EvidenceError, match='changed'):
        bf.ready_proof(tmp_path)


def test_three_empty_completed_allocations_stop_backfill(shared, monkeypatch):
    s = SimpleNamespace(config=shared.config, directory=shared.directory, runtime=shared.directory / 'runtime',
        code=Path(shared.config['code']), slurm=shared.scheduler, state={'phase': 'history', 'round_index': 1})
    bg = bf.Backfill(s)
    bg.state = {'active': 'auto_rft_0001', 'runs': {'auto_rft_0001': {
        'job_id': '99', 'stage': 'sample', 'stalls': 2, 'progress_before': 0}}}
    shared.scheduler.states['99'] = ('COMPLETED', '0:0')
    with pytest.raises(ev.EvidenceError, match='no progress'):
        bg.tick()


def test_background_keeps_running_while_foreground_waits(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    sup.atomic(path, {'root': str(tmp_path), 'code': str(tmp_path / 'code'), 'pool': {'enabled': True}})
    s = sup.Supervisor(path)
    s.state.update(phase='train_evaluate', data={'released_rows': 1}, current_run='auto_rft_0001')
    monkeypatch.setattr(s, 'check_pins', lambda: None)
    monkeypatch.setattr(s.slurm, 'queue', lambda: [])
    monkeypatch.setattr(s, 'task', lambda *args: False)
    ticks = []
    monkeypatch.setattr(bf.Backfill, 'tick', lambda self: ticks.append(True))
    s.tick()
    assert ticks == [True]
    s.directory.mkdir(parents=True, exist_ok=True)
    (s.directory / 'PAUSE').touch()
    s.tick()
    assert ticks == [True]


def report_fixture(shared):
    release = {'mean_upper': .7, 'slices': {s: {'score': .7} for s in ('v2', 'v3', 'd1')}}
    shared.config['leaderboard_snapshot'] = {'judges': {j: {'rel12e_rag': release} for j in ('gemma', 'gpt_oss')}}
    return {'updated_at': '2026-10-22T07:00:00+00:00', 'status': 'running', 'phase': 'history',
            'round_index': 1, 'decisions': []}


def test_reports_only_claim_improvement_for_completed_decisions(shared):
    state = report_fixture(shared)
    initial = reports.capture(shared.config, state, [], timestamp=state['updated_at'])
    state['tasks'] = {'training': {'status': 'running', 'job_id': '100'}}
    unchanged = reports.capture(shared.config, state, [], initial, timestamp=state['updated_at'])
    assert unchanged['improvement_percentage_points'] == 0
    assert 'No new verified score' in reports.render(unchanged)
    state['decisions'] = [{'run': 'auto_rft_0001', 'goal_met': False, 'best_observed': {
        'arm': 's43_rag', 'minimum_score': .78, 'judges': {'gemma': {'mean': .82, 'rank': 3}}}}]
    better = reports.capture(shared.config, state, [], initial, timestamp=state['updated_at'])
    assert better['improvement_percentage_points'] == pytest.approx(8)
    assert len(better['new_decisions']) == 1
    again = reports.capture(shared.config, state, [], better, timestamp=state['updated_at'])
    assert again['improvement_percentage_points'] == 0
    assert not again['new_decisions']


def test_reports_write_once_per_thirty_minutes(shared):
    state = report_fixture(shared)
    assert reports.write_report(shared.config, state, shared.scheduler, epoch=10000)
    assert reports.write_report(shared.config, state, shared.scheduler, epoch=11799) is None
    assert reports.write_report(shared.config, state, shared.scheduler, epoch=11800)
    assert len(list((shared.directory / 'reports').glob('1970*.md'))) == 2


def test_upgrade_preserves_frozen_protocol_and_rejects_existing_rounds(shared):
    shared.config.pop('pool')
    shared.config['target'] = {'minimum_score': .90}
    shared.config['pinned_files'] = {'eval': 'unchanged'}
    path = shared.directory / 'config.json'
    sup.atomic(path, shared.config)
    old_round = Path(shared.config['code']) / 'runs/auto_rft_0001'
    old_round.mkdir(parents=True)
    (old_round / 'AUTOPILOT_PLAN.json').write_text('{}')
    with pytest.raises(ev.EvidenceError, match='Existing automatic round'):
        enable_pool.migrate(path)
    assert ev.read_json(path) == shared.config


def sampling_background(shared):
    shared.config['baseline_model'] = 'release'
    runtime = shared.directory / 'runtime'
    runtime.mkdir(parents=True)
    (runtime / 'onpolicy_gen.sbatch').write_text('#!/bin/bash\n#SBATCH --nodes=8\n')
    s = SimpleNamespace(config=shared.config, directory=shared.directory, runtime=runtime,
        code=Path(shared.config['code']), slurm=shared.scheduler, state={'phase': 'train_evaluate', 'round_index': 1})
    bg = bf.Backfill(s)
    directory = s.code / 'runs/auto_rft_0001'
    jsonl(directory / 'data/questions.jsonl', [{'id': 'a'}])
    sup.atomic(directory / 'AUTOPILOT_PLAN.json', {'run': 'auto_rft_0001', 'questions': 1, 'sample_seed': 1016,
        'training_only_note': '', 'question_sha256': ev.sha256(directory / 'data/questions.jsonl')})
    job = bg.pool.submit('main', [str(runtime / 'onpolicy_gen.sbatch')], role='background', reserve=4)
    bg.state = {'active': 'auto_rft_0001', 'runs': {'auto_rft_0001': {
        'job_id': job, 'stage': 'sample', 'index': 1, 'attempt': 1, 'progress_before': 0}}}
    return bg


def test_idle_nodes_assist_current_training_data_without_duplicate_submissions(shared):
    bg = sampling_background(shared)
    bg.tick()
    assert [j['nodes'] for j in shared.scheduler.jobs] == [4, 4]
    assert bg.state['runs']['auto_rft_0001']['helper_job_id'] == '101'
    bg.tick()
    assert len(shared.scheduler.submissions) == 2


def test_judging_waits_until_both_sampling_jobs_finish(shared, monkeypatch):
    bg = sampling_background(shared)
    bg.tick()
    shared.scheduler.states['100'] = ('COMPLETED', '0:0')
    monkeypatch.setattr(bf, 'sample_proof', lambda *a: pytest.fail('helper is still writing'))
    bg.tick()
    assert bg.state['active'] == 'auto_rft_0001'


def test_no_assistance_for_future_dataset_or_judging(shared):
    bg = sampling_background(shared)
    bg.s.state['round_index'] = 2
    bg.tick()
    assert len(shared.scheduler.submissions) == 1
    bg.s.state['round_index'] = 1
    bg.state['runs']['auto_rft_0001']['stage'] = 'judge'
    bg.tick()
    assert len(shared.scheduler.submissions) == 1


def test_redundant_helper_cannot_delay_completed_sampling(shared, monkeypatch):
    bg = sampling_background(shared)
    bg.tick()
    directory = bg.code / 'runs/auto_rft_0001'
    jsonl(directory / 'gen/cands_unit_00000.jsonl', [{'id': 'a', 'candidates': ['verified sample']}])
    original = shared.scheduler.command
    cancelled = []
    def command(args):
        if args[0] != 'scancel':
            return original(args)
        cancelled.append(args[1])
        shared.scheduler.states[args[1]] = ('CANCELLED', '0:15')
        shared.scheduler.jobs = [j for j in shared.scheduler.jobs if j['id'] != args[1]]
        return ''
    monkeypatch.setattr(shared.scheduler, 'command', command)
    bg.tick()
    assert cancelled == ['101']
    assert bg.state['runs']['auto_rft_0001']['helper_cancel_requested']
    bg.tick()  # Expected cancellation does not stop the coordinator or cancel the main allocation.
    assert bg.state['active'] == 'auto_rft_0001'
