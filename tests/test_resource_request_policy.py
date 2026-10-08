"""Real generation/replay integration for the observed request-parser failures.

FakeTeacher supplies synthetic responses; tools execute local deterministic mocks
only. These are policy regressions, not real-model correctness measurements.
"""
import copy
import hashlib
import json

import pytest

from distillkit import generate, tools, toolcheck, verify
from distillkit.config import Config
from distillkit.io import read_jsonl, write_jsonl
from distillkit.job_status_guard import (
    JobStatusPolicy, LEGACY_POLICY_VERSION, POLICY_VERSION, WORKFLOW_VERSION,
)
from distillkit.teacher import Completion
from test_job_status_workflow import FakeTeacher, call, fixture
from test_pipeline import _cfg_dict
from test_scenario_plan import planned, fake_teacher
from test_source_registry import setup


@pytest.fixture
def cfg(tmp_path):
    data = _cfg_dict(tmp_path)
    data['generate'].update(gold_dir=None, max_tool_rounds=4, job_status_policy_version=POLICY_VERSION)
    return Config.model_validate(data)


def run(cfg, question, batches, *, catalog=None, version=POLICY_VERSION):
    teacher = FakeTeacher([*(Completion('I will check.', batch) for batch in batches),
                           Completion('The requested observations were returned.')])
    policy, audit = JobStatusPolicy(version=version), {}
    transcript = generate.tool_trace(
        teacher, cfg, 'Synthetic simulator contract; no real cluster.', question,
        mock_job_ids=catalog, tool_policy=policy, audit=audit,
    )
    row = {'id':'parser-regression', 'mode':'call', 'question':question, 'messages':transcript,
           'tools':tools.SCHEMAS, 'mock_version':tools.MOCK_VERSION,
           'tool_policy':policy.as_dict(), 'runtime_audit':audit,
           'workflow_version':WORKFLOW_VERSION}
    if catalog is not None:
        row.update(mock_job_ids=catalog, mock_catalog_version=tools.CATALOG_VERSION)
    return row, teacher


def test_old_config_and_direct_callers_keep_legacy_defaults(tmp_path):
    data = _cfg_dict(tmp_path)
    assert 'job_status_policy_version' not in data['generate']
    cfg = Config.model_validate(data)
    assert cfg.generate.job_status_policy_version == LEGACY_POLICY_VERSION
    assert JobStatusPolicy().version == LEGACY_POLICY_VERSION
    data['generate']['job_status_policy_version'] = 'unknown-policy'
    with pytest.raises(ValueError, match='job_status_policy_version'):
        Config.model_validate(data)


@pytest.mark.parametrize('version', [LEGACY_POLICY_VERSION, POLICY_VERSION])
def test_real_generation_resolves_explicit_policy_version(cfg, tmp_path, monkeypatch, version):
    cfg.generate.job_status_policy_version = version
    cfg.generate.job_status_discovery = 'clarify_first'
    cfg.generate.personas = ['a synthetic test user']
    cfg.generate.task_types = ['howto']
    cfg.generate.questions_per_chunk = cfg.generate.answers_per_question = 1
    cfg.generate.tool_fraction = 1
    cfg.generate.tool_mix = {'call': 1}
    cfg.seeds.dir = tmp_path / 'seeds'
    cfg.seeds.dir.mkdir()
    (cfg.seeds.dir / 'fixture.md').write_text('Synthetic source: partitions have memory limits.')

    class PipelineTeacher(FakeTeacher):
        model = 'fake-parser-teacher-no-inference'
        def map(self, fn, items):
            return [fn(item) for item in items]
        def chat_json(self, system, prompt, schema):
            return schema(question='Check the memory limits of every partition.'), 'synthetic fixture'

    teacher = PipelineTeacher([Completion('I will check.', [call('partition_info')]),
                               Completion('The returned limits are observations.')])
    monkeypatch.setattr(generate, 'Teacher', lambda _: teacher)
    generate.run(cfg)
    rows = read_jsonl(cfg.run_dir / 'generated.jsonl')
    assert len(rows) == 1
    row = rows[0]
    assert row['tool_policy']['version'] == version
    assert row['runtime_audit']['policy'] == row['tool_policy']
    check = toolcheck.check_trace(row)
    assert check['runtime_enforcement_consistent']
    assert check['passed'] is (version == POLICY_VERSION)
    assert all(f'Application policy {version}:' in r['messages'][0]['content'] for r in teacher.requests)


@pytest.mark.parametrize('version', [LEGACY_POLICY_VERSION, POLICY_VERSION])
def test_direct_bounded_workflow_resolves_version_and_discovery(cfg, version):
    cfg.generate.job_status_policy_version = version
    cfg.generate.job_status_discovery = 'allow_readonly'
    audit = {}
    teacher = FakeTeacher([Completion('I will list candidates.', [call('list_queue')]),
                           Completion('Which returned job is yours?')])
    messages = generate.tool_trace(teacher, cfg, 'Synthetic catalog source.', 'Check my job.',
                                   job_status_workflow=True, audit=audit)
    assert audit['policy'] == {'discovery': 'allow_readonly', 'version': version}
    assert audit['response_origin'] == 'teacher' and len(teacher.requests) == 2
    row = {'question': 'Check my job.', 'mode': 'call', 'messages': messages,
           'mock_version': tools.MOCK_VERSION, 'tool_policy': audit['policy'],
           'runtime_audit': audit, 'workflow_version': WORKFLOW_VERSION}
    assert toolcheck.check_trace(row)['passed']


def test_policy_version_change_cannot_resume_scenario_run(planned, monkeypatch):
    cfg, _, _ = planned
    assert cfg.generate.job_status_policy_version == LEGACY_POLICY_VERSION
    fake_teacher(cfg, monkeypatch)
    generate.run(cfg)
    before = (cfg.run_dir / 'generated.jsonl').read_bytes()
    cfg.generate.job_status_policy_version = POLICY_VERSION
    monkeypatch.setattr(generate, 'Teacher', lambda *_: pytest.fail('client created before changed-policy refusal'))
    with pytest.raises(ValueError, match='fresh run directory'):
        generate.run(cfg)
    assert (cfg.run_dir / 'generated.jsonl').read_bytes() == before


# Actual observed wording shapes, with no teacher answers copied into fixtures.
CASES = [
    ('memory-capacity',
     "How can I check the memory limits of the 'dcgp_usr_prod' partition and the current availability of free GPUs in that partition?",
     [[call('partition_info', partition='dcgp_usr_prod'),
       call('gpu_availability', partition='dcgp_usr_prod')]]),
    ('allocation-and-free-pool',
     'How can I check how many GPUs are allocated to my running job 4726664 and how many are currently free in its partition?',
     [[call(job_id='4726664')], [call('gpu_availability', partition='boost_usr_prod')]]),
    ('job-and-partition-limits',
     'How can I check the time limit of job 4487316 and compare it to the maximum wall time allowed by its partition?',
     [[call(job_id='4487316')], [call('partition_info', partition='boost_qos_dbg')]]),
    ('job-with-id',
     'I want to check the status of my job with ID 4374554. Also, can you list all my pending and running jobs?',
     [[call(job_id='4374554')], [call('list_queue')]]),
    ('capacity-screen',
     'How can I check which partitions have nodes with at least 4 GPUs per node and also see how many free GPUs are currently available in those partitions?',
     [[call('partition_info')], [call('gpu_availability')]]),
    ('all-partition-views',
     'How can I list all partition limits and current GPU availability per partition, and reconcile the partition names in both views?',
     [[call('partition_info'), call('gpu_availability')]]),
    ('pending-and-availability',
     'Can you list the pending jobs in boost_usr_prod and check the current GPU availability for that partition?',
     [[call('list_queue', state='PENDING', partition='boost_usr_prod')],
      [call('gpu_availability', partition='boost_usr_prod')]]),
]


@pytest.mark.parametrize('name,question,batches', CASES, ids=[case[0] for case in CASES])
def test_observed_requests_reach_real_prompt_loop_and_replay(cfg, name, question, batches):
    row, teacher = run(cfg, question, batches)
    checked = toolcheck.check_trace(row)
    assert checked['passed'] and checked['runtime_enforcement_consistent']
    assert checked['guard_blocked_calls'] == 0
    assert checked['executed_calls'] == sum(len(batch) for batch in batches)
    assert row['runtime_audit']['response_origin'] == 'teacher'
    assert len(teacher.requests) == len(batches) + 1
    assert all(f'Application policy {POLICY_VERSION}:' in r['messages'][0]['content']
               for r in teacher.requests)
    assert all(generate.PROMPT_VERSION in r['messages'][0]['content'] for r in teacher.requests)
    assert teacher.requests[-1]['messages'][-1]['role'] == 'tool'


@pytest.mark.parametrize('name,question,batches', CASES, ids=[case[0] for case in CASES])
def test_archived_v2_keeps_old_blocks_and_exact_audit_replay(cfg, name, question, batches):
    row, _ = run(cfg, question, batches, version=LEGACY_POLICY_VERSION)
    checked = toolcheck.check_trace(row)
    assert not checked['passed'] and checked['application_policy_errors']
    assert checked['runtime_enforcement_consistent']
    assert checked['guard_blocked_calls'] > 0
    assert row['runtime_audit']['model_final_answer'] is None
    assert row['runtime_audit']['response_origin'] == 'runtime_guard'
    # Re-labelling an archived attempt cannot make its blocked results into data.
    changed = copy.deepcopy(row)
    changed['tool_policy']['version'] = POLICY_VERSION
    assert not toolcheck.check_trace(changed)['passed']


@pytest.mark.parametrize('question,proposed', [
    ('Check the current GPU availability.', call('partition_info')),
    ('Check the maximum wall time and memory capacity of every partition.', call('gpu_availability')),
    ('Explain how to check the limits of a partition.', call('partition_info')),
    ('Can you describe how to check free GPUs?', call('gpu_availability')),
    ('Show an example of how to check free GPUs.', call('gpu_availability')),
    ('How do I check partition limits without using tools?', call('partition_info')),
    ('What command can I use to check GPU availability?', call('gpu_availability')),
    ('Which commands show current GPU availability?', call('gpu_availability')),
    ('Show memory limits for the partition; the source also describes current GPU availability.', call('gpu_availability')),
    ('Check job 7100001; it was allocated 4 GPUs.', call('gpu_availability')),
    ('The source mentions partition limits. Check job 7100001.', call('partition_info')),
    ('Show my available modules.', call('gpu_availability')),
    ('Do not check the current GPU availability.', call('gpu_availability')),
    ("Don't list all partition limits.", call('partition_info')),
    ('Never look up partition limits.', call('partition_info')),
    ('Ask me before checking which partitions have 4 GPUs per node.', call('partition_info')),
    ('Do not check job with ID 7100001.', call(job_id='7100001')),
])
def test_capability_scope_explanation_and_negation_still_block_before_execution(cfg, question, proposed, monkeypatch):
    executed = []
    monkeypatch.setattr(generate, 'execute', lambda *args: executed.append(args))
    row, teacher = run(cfg, question, [[proposed]], catalog=['7100001'])
    assert not executed and len(teacher.requests) == 1
    checked = toolcheck.check_trace(row)
    assert checked['guard_blocked_calls'] == 1 and checked['runtime_enforcement_consistent']


def test_live_question_about_free_gpu_count_still_passes(cfg):
    row, _ = run(cfg, 'Can you check how many GPUs are free?',
                 [[call('gpu_availability')]])
    assert toolcheck.check_trace(row)['passed']
    assert row['runtime_audit']['response_origin'] == 'teacher'


@pytest.mark.parametrize('question,expected', [
    ('Check my job with ID 7100001.', '7100001'),
    ('Check the job with the number: "7100001".', '7100001'),
    ('Check job ID 7100001.', '7100001'),
])
def test_supported_explicit_identifier_forms(cfg, question, expected):
    row, _ = run(cfg, question, [[call(job_id=expected)]], catalog=[expected])
    assert toolcheck.check_trace(row)['passed']


@pytest.mark.parametrize('question', [
    'Check my job with 8 GPUs for 30 minutes.',
    'I lost my job ID. It requested 8 GPUs.',
    'Check my job with ID 7100001 or my job with ID 7100002; I have not selected one.',
])
def test_missing_or_ambiguous_identifiers_still_block(cfg, question, monkeypatch):
    executed = []
    monkeypatch.setattr(generate, 'execute', lambda *args: executed.append(args))
    row, _ = run(cfg, question, [[call(job_id='7100001')]], catalog=['7100001','7100002'])
    checked = toolcheck.check_trace(row)
    assert not executed and checked['guard_blocked_calls'] == 1
    assert checked['runtime_enforcement_consistent']
    if '7100002' in question:
        assert row['runtime_audit']['user_response'] == 'Which job ID should I inspect?'


def test_partition_filter_requires_user_or_completed_prior_result(cfg, monkeypatch):
    question = 'Check job 4726664 and how many GPUs are currently free in its partition.'
    # The partition is in the fixture, but this authored batch has not observed it.
    executed = []
    monkeypatch.setattr(generate, 'execute', lambda *args: executed.append(args))
    row, _ = run(cfg, question, [[call(job_id='4726664'),
                                 call('gpu_availability', partition='boost_usr_prod')]])
    checked = toolcheck.check_trace(row)
    assert not executed and checked['guard_blocked_calls'] == 2
    assert checked['runtime_enforcement_consistent']
    assert any('Partition filter' in e for e in checked['application_policy_errors'])


def test_observed_partition_can_be_used_only_in_later_turn(cfg):
    question = 'Check job 4726664 and how many GPUs are currently free in its partition.'
    row, teacher = run(cfg, question, [[call(job_id='4726664')],
                                      [call('gpu_availability', partition='boost_usr_prod')]])
    assert toolcheck.check_trace(row)['passed']
    prior = json.loads(teacher.requests[1]['messages'][-1]['content'])
    assert prior['partition'] == 'boost_usr_prod'


def test_wrong_partition_filter_does_not_use_system_source_or_gold(cfg, monkeypatch):
    executed = []
    monkeypatch.setattr(generate, 'execute', lambda *args: executed.append(args))
    row, _ = run(cfg, 'Check free GPUs in boost_usr_prod.',
                 [[call('gpu_availability', partition='boost_qos_dbg')]])
    assert not executed and toolcheck.check_trace(row)['guard_blocked_calls'] == 1


def test_log_text_or_error_partition_cannot_supply_filter():
    from distillkit.job_status_guard import policy_error
    proposed = {'name':'gpu_availability','arguments':{'partition':'boost_usr_prod'}}
    policy = JobStatusPolicy(version=POLICY_VERSION)
    question = 'Check current free GPUs.'
    for result in [{'lines':['partition boost_usr_prod']},
                   {'partition':'boost_usr_prod','error':'not found'},
                   {'partition':'boost_usr_prod','found':False}]:
        assert policy_error(proposed, question, [result], policy)
    assert policy_error(proposed, question, [{'jobs':[{'partition':'boost_usr_prod'}]}], policy) is None


def test_queue_result_job_requires_prior_turn_and_explicit_selection(cfg, monkeypatch):
    jid = fixture('RUNNING')
    question = 'List my running jobs and inspect the only result.'
    row, _ = run(cfg, question, [[call('list_queue',state='RUNNING')], [call(job_id=jid)]], catalog=[jid])
    assert toolcheck.check_trace(row)['passed']
    executed=[]
    monkeypatch.setattr(generate, 'execute', lambda *args: executed.append(args))
    row, _ = run(cfg, question, [[call('list_queue',state='RUNNING'),call(job_id=jid)]], catalog=[jid])
    assert not executed and toolcheck.check_trace(row)['guard_blocked_calls'] == 2


def test_production_verify_accepts_new_policy_replay_but_still_requires_support(cfg, tmp_path):
    source = 'Synthetic simulator: partition limits are distinct from current availability.'
    cfg.seeds.dir = tmp_path/'seeds'
    cfg.seeds.dir.mkdir()
    (cfg.seeds.dir/'fixture.md').write_text(source)
    cfg.generate.job_status_discovery = 'clarify_first'
    row, _ = run(cfg, 'Check all partition limits and current GPU availability.',
                 [[call('partition_info'),call('gpu_availability')]])
    row.update(doc_id='fixture',source_kind='full',source_sha256=hashlib.sha256(source.encode()).hexdigest())
    write_jsonl(cfg.run_dir/'generated.jsonl',[row])
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir/'verified.jsonl') == []
    assert read_jsonl(cfg.run_dir/'rejected.jsonl') == []
    pending = read_jsonl(cfg.run_dir/'pending_review.jsonl')
    assert len(pending) == 1 and pending[0]['tool_checks']['passed']
    assert pending[0]['pending_reason'] == 'grounding_review_unresolved'


@pytest.mark.parametrize('configured,recorded', [
    (POLICY_VERSION, LEGACY_POLICY_VERSION), (LEGACY_POLICY_VERSION, POLICY_VERSION),
])
def test_config_does_not_silently_accept_other_policy_rows(cfg, configured, recorded):
    cfg.generate.job_status_discovery = 'clarify_first'
    cfg.generate.job_status_policy_version = configured
    row, _ = run(cfg, 'Show the current free GPUs.', [[call('gpu_availability')]],
                 version=recorded)
    assert toolcheck.check_trace(row)['passed']  # each row is replayable under its own policy
    write_jsonl(cfg.run_dir/'generated.jsonl',[row])
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir/'rejected.jsonl')[0]['reject_reason'] == 'application_policy_mismatch'
