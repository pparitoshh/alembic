"""Deployment policy integration; expected labels never select runtime behavior."""
import copy
import json

import pytest

from distillkit import generate, toolcheck, tools, verify
from distillkit.config import Config
from distillkit.io import read_jsonl, write_jsonl
from distillkit.job_status_guard import JobStatusPolicy, WORKFLOW_VERSION
from distillkit.records import training_example
from distillkit.teacher import Completion
from test_job_status_workflow import FakeTeacher, call, fixture
from test_pipeline import _cfg_dict


@pytest.fixture
def cfg(tmp_path):
    data = _cfg_dict(tmp_path)
    data['generate'].update(gold_dir=None, max_tool_rounds=4)
    return Config.model_validate(data)


def run(cfg, teacher, question, catalog, discovery='clarify_first', **metadata):
    policy, audit = JobStatusPolicy(discovery), {}
    convo = generate.tool_trace(teacher, cfg, 'Synthetic source: status is not a diagnosis.', question,
                               mock_job_ids=catalog, tool_policy=policy, audit=audit)
    return dict(id='diagnostic', mode='ask', question=question, messages=convo, tools=tools.SCHEMAS,
                mock_version=tools.MOCK_VERSION, mock_catalog_version=tools.CATALOG_VERSION,
                mock_job_ids=catalog, tool_policy=policy.as_dict(), runtime_audit=audit,
                workflow_version=WORKFLOW_VERSION, **metadata)


def test_observed_filename_lookup_is_blocked_before_execution_and_kept_separate(cfg, monkeypatch):
    question='Please check whether the submission from my batch script `sweep.sh` is still running. I only have the script filename, not a scheduler job ID.'
    executed=[]
    monkeypatch.setattr(generate, 'execute', lambda *args: executed.append(args))
    teacher=FakeTeacher([Completion('I will check for running jobs associated with your script name.',
                                     [call('list_queue', state='RUNNING')])])
    row=run(cfg, teacher, question, [fixture('RUNNING')])
    assert executed==[] and len(teacher.requests)==1
    assert row['messages'][1]['tool_calls'][0]['function']['name']=='list_queue'
    audit=row['runtime_audit']
    assert audit['events'][0]['blocked'] and not audit['events'][0]['executed']
    assert audit['model_final_answer'] is None and audit['response_origin']=='runtime_guard'
    assert audit['user_response']==row['messages'][-1]['content']
    check=toolcheck.check_trace(row)
    assert not check['model_policy_compliant'] and not check['passed']
    assert check['guard_blocked_calls']==1 and check['executed_calls']==0
    assert check['runtime_enforcement_consistent']
    write_jsonl(cfg.run_dir/'generated.jsonl',[row]);verify.run(cfg)
    assert read_jsonl(cfg.run_dir/'verified.jsonl')==[]
    assert read_jsonl(cfg.run_dir/'rejected.jsonl')[0]['reject_reason']=='application_policy_violation'
    with pytest.raises(ValueError, match='application output'):
        training_example(row, 'system')


def test_same_missing_id_request_can_discover_then_clarify_under_explicit_policy(cfg):
    jid=fixture('RUNNING')
    teacher=FakeTeacher([Completion('I will list candidates.',[call('list_queue')]),
                         Completion('Which listed job is the one you want checked?')])
    row=run(cfg,teacher,'I only know my script filename; can you find my job?', [jid], 'allow_readonly')
    check=toolcheck.check_trace(row)
    assert check['passed'] and check['model_policy_compliant'] and check['executed_calls']==1
    assert row['runtime_audit']['response_origin']=='teacher'


@pytest.mark.parametrize('discovery',['clarify_first','allow_readonly'])
def test_explicit_queue_and_resource_queries_need_no_job_id(cfg, discovery):
    for question, proposed in [('List my pending jobs.',call('list_queue',state='PENDING')),
                               ('List my jobs; I previously saw job 7100001 and job 7100002.',call('list_queue')),
                               ('Show the current free GPUs.',call('gpu_availability')),
                               ('List partition limits.',call('partition_info'))]:
        row=run(cfg,FakeTeacher([Completion('',[proposed]),Completion('Here are the returned fields.')]),
                question,[],discovery)
        assert toolcheck.check_trace(row)['passed']


@pytest.mark.parametrize('state',['RUNNING','PENDING','FAILED','COMPLETED'])
def test_identified_job_call_remains_supported(cfg,state):
    jid=fixture(state)
    row=run(cfg,FakeTeacher([Completion('',[call(job_id=jid)]),Completion(f'The state is {state}.')]),
            f'Check job {jid}.',[jid])
    assert toolcheck.check_trace(row)['passed']
    assert json.loads(row['messages'][2]['content'])['state']==state


def test_unknown_id_returns_actual_not_found_instead_of_guard_rewrite(cfg):
    row=run(cfg,FakeTeacher([Completion('',[call(job_id='7199999')]),Completion('Not found in the simulated catalog.')]),
            'Check job 7199999.',[])
    assert toolcheck.check_trace(row)['passed']
    assert json.loads(row['messages'][2]['content'])['found'] is False
    assert row['runtime_audit']['response_origin']=='teacher'


@pytest.mark.parametrize('question,jid',[
    ('I requested 8 GPUs for 30 minutes. Check my job.','8'),
    ('I have job 7100001 and job 7100002. Which should I check?','7100001'),
    ('Do not look up job 7100001; ask me before tools.','7100001')])
def test_missing_ambiguous_or_forbidden_target_is_not_executed(cfg,question,jid):
    row=run(cfg,FakeTeacher([Completion('',[call(job_id=jid)])]),question,[jid])
    check=toolcheck.check_trace(row)
    assert check['executed_calls']==0 and check['guard_blocked_calls']==1
    assert not check['passed']


def test_model_clarification_is_not_counted_as_guard_intervention(cfg):
    row=run(cfg,FakeTeacher([Completion('What is your job ID?')]),'Check my job.',[])
    check=toolcheck.check_trace(row)
    assert check['passed'] and check['guard_blocked_calls']==0
    assert row['runtime_audit']['model_final_answer']=='What is your job ID?'


def test_one_discovered_candidate_does_not_resolve_script_association(cfg):
    jid=fixture('RUNNING')
    row=run(cfg,FakeTeacher([Completion('',[call('list_queue')]),Completion('',[call(job_id=jid)])]),
            'Please find the job for my script.',[jid],'allow_readonly')
    check=toolcheck.check_trace(row)
    assert check['executed_calls']==1 and check['guard_blocked_calls']==1
    assert not check['passed'] and check['runtime_enforcement_consistent']


def test_explicit_inspection_of_only_result_uses_executed_prior_result(cfg):
    jid=fixture('RUNNING')
    row=run(cfg,FakeTeacher([Completion('',[call('list_queue')]),Completion('',[call(job_id=jid)]),
                            Completion('The discovered job is RUNNING.')]),
            'List my running jobs and inspect the only result.',[jid])
    assert toolcheck.check_trace(row)['passed']


def test_simultaneous_discovery_cannot_authorize_job_call(cfg,monkeypatch):
    jid=fixture('RUNNING');executed=[]
    monkeypatch.setattr(generate,'execute',lambda *args:executed.append(args))
    row=run(cfg,FakeTeacher([Completion('',[call('list_queue'),call(job_id=jid)])]),
            'List my running jobs and inspect the only result.',[jid])
    assert executed==[]
    assert toolcheck.check_trace(row)['guard_blocked_calls']==2


def test_hidden_metadata_does_not_change_runtime_decisions(cfg):
    def one():
        return run(cfg,FakeTeacher([Completion('',[call('list_queue')])]),'Check my job.',[])
    a,b=one(),one()
    a.update(id='ask-gold',mode='ask',expected_job_id=None)
    b.update(id='call-gold',mode='call',expected_job_id='9999999',reference='The job is running.')
    assert a['messages']==b['messages'] and a['runtime_audit']==b['runtime_audit']
    assert not toolcheck.check_trace(a)['model_policy_compliant']
    assert not toolcheck.check_trace(b)['model_policy_compliant']


def test_tampered_audit_and_diagnostic_promotion_rejected(cfg):
    row=run(cfg,FakeTeacher([Completion('What is the job ID?')]),'Check my job.',[])
    changed=copy.deepcopy(row);changed['runtime_audit']['user_response']='It is running.'
    assert toolcheck.check_trace(changed)['trace_errors']
    row.update(purpose='diagnostic_only',training_eligible=False)
    with pytest.raises(ValueError,match='not training'):
        training_example(row,'system')


@pytest.mark.parametrize('function',[None, [], {'name':[]}])
def test_malformed_policy_call_is_per_record_rejection(cfg,function):
    row=run(cfg,FakeTeacher([Completion('What is the job ID?')]),'Check my job.',[])
    row['messages'][1]['tool_calls']=[{'function':function}]
    check=toolcheck.check_trace(row)
    assert not check['passed'] and check['call_errors']
    assert not check['runtime_enforcement_consistent']


@pytest.mark.parametrize('change',['content','origin','extra_turn'])
def test_guard_delivery_is_reconstructed_not_trusted_from_audit(cfg,change,monkeypatch):
    row=run(cfg,FakeTeacher([Completion('',[call('list_queue')])]),'Check my job.',[])
    if change=='content':
        row['messages'][-1]['content']='I cancelled your job successfully.'
        row['runtime_audit']['user_response']=row['messages'][-1]['content']
    elif change=='origin':
        row['messages'][-1].pop('origin')
    else:
        row['messages'].extend([{'role':'assistant','content':'','tool_calls':[call(job_id='7100001')]}])
    executed=[]
    monkeypatch.setattr(toolcheck,'execute',lambda *args:executed.append(args))
    check=toolcheck.check_trace(row)
    assert not executed and check['trace_errors'] and not check['runtime_enforcement_consistent']


@pytest.mark.parametrize('question,proposed',[
    ('Do not list my jobs; ask me for my job ID instead.',call('list_queue')),
    ('Never show my queue.',call('list_queue')),
    ("Don't check job 7100001; ask me first.",call(job_id='7100001')),
    ('Do not find my jobs yet.',call('list_queue')),
    ('Never search my queue.',call('list_queue')),
    ('Ask me before checking job 7100001.',call(job_id='7100001'))])
def test_explicit_negative_lookup_verbs_take_precedence(cfg,question,proposed,monkeypatch):
    executed=[]
    monkeypatch.setattr(generate,'execute',lambda *args:executed.append(args))
    row=run(cfg,FakeTeacher([Completion('',[proposed])]),question,['7100001'])
    assert not executed and toolcheck.check_trace(row)['guard_blocked_calls']==1
