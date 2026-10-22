"""Regression coverage for completed jobs purged from squeue and safe controller repair."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research/autopilot'))
import evidence as ev
import supervisor as sup
import wait_gpu
import upgrade_waiter


@pytest.fixture
def config(tmp_path):
    path = tmp_path / 'ctl/autopilot/config.json'
    sup.atomic(path, {'root': str(tmp_path), 'code': str(tmp_path / 'code'), 'python': sys.executable})
    return path


@pytest.mark.parametrize('state', ['TIMEOUT', 'COMPLETED', 'NODE_FAIL', 'OUT_OF_MEMORY'])
def test_finished_job_uses_accounting_when_no_longer_in_live_queue(config, monkeypatch, state):
    def command(self, args, **kwargs):
        assert args[0] == 'sacct', 'squeue -j can reject IDs already purged from its live cache'
        return '123|' + state + '|0:0\n'
    monkeypatch.setattr(sup.Slurm, 'command', command)
    monkeypatch.setattr(wait_gpu.time, 'sleep', lambda _: pytest.fail('Terminal accounting must release the waiter'))
    assert wait_gpu.wait(config, '123', config.parent / 'HEARTBEAT') == state
    assert (config.parent / 'HEARTBEAT').exists()


def test_delayed_accounting_and_scheduler_failure_do_not_count_as_completion(config, monkeypatch):
    replies = iter([sup.SchedulerError('offline'), ('UNKNOWN', None), ('RUNNING', '0:0'), ('TIMEOUT', '0:0')])
    sleeps = []
    def status(self, job):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(sup.Slurm, 'status', status)
    monkeypatch.setattr(wait_gpu.time, 'sleep', sleeps.append)
    assert wait_gpu.wait(config, '123', config.parent / 'HEARTBEAT') == 'TIMEOUT'
    assert sleeps == [30, 30, 30]


def test_waiter_respects_cancellation_and_stop(config, monkeypatch):
    monkeypatch.setattr(sup.Slurm, 'status', lambda *args: ('CANCELLED', '0:15'))
    with pytest.raises(ev.EvidenceError, match='cancelled'):
        wait_gpu.wait(config, '123', config.parent / 'HEARTBEAT')
    (config.parent / 'STOP').touch()
    monkeypatch.setattr(sup.Slurm, 'status', lambda *args: pytest.fail('Do not query after STOP'))
    with pytest.raises(ev.EvidenceError, match='STOP'):
        wait_gpu.wait(config, '123', config.parent / 'HEARTBEAT')


def prepared(config):
    root = Path(ev.read_json(config)['root'])
    script = root / 'ctl/auto_rft_0001/controller.sh'
    script.parent.mkdir(parents=True)
    script.write_text('#!/bin/bash\n' + upgrade_waiter.OLD_BLOCK + 'echo preserved\n')
    plan = root / 'code/runs/auto_rft_0001/AUTOPILOT_PLAN.json'
    sup.atomic(plan, {'run': 'auto_rft_0001', 'controller_sha256': ev.sha256(script),
                     'sample_seed': 1016, 'question_sha256': 'unchanged', 'acceptance_score_vs_teacher': .75})
    (config.parent / 'PAUSE').touch()
    return script, plan


def test_upgrade_preserves_experiment_inputs_and_is_repeatable(config):
    script, plan = prepared(config)
    original_script, original_plan = script.read_text(), ev.read_json(plan)
    assert len(upgrade_waiter.migrate(config)) == 1
    revised = ev.read_json(plan)
    assert revised['controller_sha256'] == ev.sha256(script)
    assert {k: v for k, v in revised.items() if k != 'controller_sha256'} == {
        k: v for k, v in original_plan.items() if k != 'controller_sha256'}
    assert 'wait_gpu.py' in script.read_text() and script.read_text().endswith('echo preserved\n')
    backup = config.parent / 'repairs/accounting-wait-v1/auto_rft_0001'
    assert (backup / 'controller.sh').read_text() == original_script
    assert ev.read_json(backup / 'AUTOPILOT_PLAN.json') == original_plan
    assert upgrade_waiter.migrate(config) == []


def test_upgrade_recovers_after_script_changed_before_plan_is_saved(config):
    script, plan = prepared(config)
    original_plan = ev.read_json(plan)
    upgrade_waiter.migrate(config)
    sup.atomic(plan, original_plan)
    assert len(upgrade_waiter.migrate(config)) == 1
    assert ev.read_json(plan)['controller_sha256'] == ev.sha256(script)


def test_upgrade_rejects_unexpected_script_change_and_requires_pause(config):
    script, plan = prepared(config)
    original_plan = plan.read_text()
    script.write_text(script.read_text() + 'echo unexpected\n')
    with pytest.raises(ev.EvidenceError, match='changed'):
        upgrade_waiter.migrate(config)
    assert plan.read_text() == original_plan
    (config.parent / 'PAUSE').unlink()
    with pytest.raises(ev.EvidenceError, match='Pause'):
        upgrade_waiter.migrate(config)
