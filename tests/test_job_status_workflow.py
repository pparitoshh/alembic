"""Deterministic diagnostics exercise production tool_trace, not a substitute model run."""
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from distillkit import generate, tools, toolcheck
from distillkit.config import Config
from distillkit.teacher import Completion
from test_pipeline import _cfg_dict


@pytest.fixture
def cfg(tmp_path):
    d = _cfg_dict(tmp_path)
    d['generate'].update(gold_dir=None, max_tool_rounds=3)
    return Config.model_validate(d)


def call(name='job_status', **args):
    return {'id':'c1','type':'function','function':{'name':name,'arguments':json.dumps(args)}}


class FakeTeacher:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
    def complete(self, messages, tools):
        self.requests.append({'messages':messages,'tools':tools})
        return next(self.responses)


def fixture(state):
    return next(str(j) for j in range(7100000,7101000) if tools.execute('job_status',{'job_id':str(j)})['state']==state)


def trace(cfg, teacher, question, catalog, mode='call'):
    convo = generate.tool_trace(teacher,cfg,'Synthetic simulator contract; no real cluster.',question,mock_job_ids=catalog)
    row = {'mode':mode,'messages':convo,'mock_version':tools.MOCK_VERSION,
           'mock_catalog_version':tools.CATALOG_VERSION,'mock_job_ids':catalog}
    return row


@pytest.mark.parametrize('state',['RUNNING','PENDING','COMPLETED','FAILED'])
def test_diagnostic_status_fields_and_real_prompt_loop(cfg,state):
    jid=fixture(state); result=tools.execute('job_status',{'job_id':jid})
    teacher=FakeTeacher([Completion('I will check that job.',[call(job_id=jid)]),
                         Completion(f"Job {jid}: {state}."+(f" Reason: {result['reason']}." if state=='PENDING' else ''))])
    row=trace(cfg,teacher,f'Please check the current state of job {jid}.',[jid])
    assert toolcheck.check_trace(row)['passed']
    assert json.loads(row['messages'][2]['content'])==result
    assert len(teacher.requests)==2
    assert all(r['tools']==tools.SCHEMAS for r in teacher.requests)
    assert teacher.requests[-1]['messages'][-1]['role']=='tool'
    assert json.loads(teacher.requests[-1]['messages'][-1]['content'])==result
    assert 'not-found' in teacher.requests[0]['messages'][0]['content']


def test_unknown_valid_job_returns_real_error_contract_without_invented_state(cfg):
    jid='7199999'
    teacher=FakeTeacher([Completion('I will look up that ID.',[call(job_id=jid)]),Completion('That job was not found in the simulated catalog. Please check its ID.')])
    row=trace(cfg,teacher,f'Check job {jid}.',[])
    result=json.loads(row['messages'][2]['content'])
    assert result=={'job_id':jid,'found':False,'error':f'job {jid} not found in this simulated job catalog'}
    assert 'state' not in result and toolcheck.check_trace(row)['passed']


@pytest.mark.parametrize('mode,question,answer',[
    ('ask','Check my job; it requested 8 GPUs for 30 minutes.','What is the job ID?'),
    ('none','What information can job_status return?','It can return state, partition, elapsed time and the job time limit.')])
def test_clarification_and_general_question_do_not_call(cfg,mode,question,answer):
    teacher=FakeTeacher([Completion(answer)])
    row=trace(cfg,teacher,question,[],mode)
    assert len(row['messages'])==2 and toolcheck.check_trace(row)['passed']
    assert teacher.requests[0]['tools']==tools.SCHEMAS


@pytest.mark.parametrize('question,jid',[
    ('Check my job; it requested 8 GPUs for 30 minutes.','8'),
    ('Check my job.','1234')])
def test_old_unrelated_number_or_guessed_id_never_reaches_mock(cfg,monkeypatch,question,jid):
    executed=[]
    monkeypatch.setattr(generate,'execute',lambda *a: executed.append(a))
    teacher=FakeTeacher([Completion('',[call(job_id=jid)]),Completion('What is the job ID?')])
    row=trace(cfg,teacher,question,[jid])
    assert executed==[]  # Old production path executed the guessed numeric fixture.
    assert 'not an explicitly identified job' in json.loads(row['messages'][2]['content'])['error']
    assert not toolcheck.check_trace(row)['passed']  # Retain failed attempt; never silently approve recovery.


def test_id_from_real_queue_result_is_usable_and_catalog_consistent(cfg):
    jid=fixture('RUNNING')
    teacher=FakeTeacher([Completion('',[call('list_queue')]),Completion('',[call(job_id=jid)]),Completion('The discovered job is RUNNING.')])
    row=trace(cfg,teacher,'List my running jobs and inspect the only result.',[jid])
    assert toolcheck.check_trace(row)['passed']
    results=[json.loads(m['content']) for m in row['messages'] if m['role']=='tool']
    assert results[0]['jobs'][0]['job_id']==results[1]['job_id']==jid
    assert results[0]['jobs'][0]['state']==results[1]['state']=='RUNNING'


def test_catalog_isolation_across_concurrent_production_conversations(cfg):
    jid=fixture('RUNNING')
    def one(catalog):
        t=FakeTeacher([Completion('',[call(job_id=jid)]),Completion('Result received.')])
        row=trace(cfg,t,f'Check job {jid}.',catalog)
        assert toolcheck.check_trace(row)['passed']
        return json.loads(row['messages'][2]['content'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        known,unknown=list(pool.map(one,[[jid],[]]))
    assert known['state']=='RUNNING' and unknown['found'] is False
    assert tools.execute('job_status',{'job_id':jid})==known  # Default remains compatible.


def test_catalog_replay_requires_version_and_rejects_mismatched_results(cfg):
    jid=fixture('RUNNING')
    row=trace(cfg,FakeTeacher([Completion('',[call(job_id=jid)]),Completion('Running.')]),f'Check job {jid}.',[jid])
    row['mock_job_ids']=[]
    assert toolcheck.check_trace(row)['result_errors']
    row.pop('mock_catalog_version')
    result=toolcheck.check_trace(row)
    assert not result['passed'] and 'version' in result['trace_errors'][0]


@pytest.mark.parametrize('ids',[['7100','7100'],['bogus'],{'7100':True}])
def test_catalog_malformed_state_rejected_before_teacher(cfg,ids):
    teacher=FakeTeacher([])
    with pytest.raises(ValueError,match='unique numeric'):
        generate.tool_trace(teacher,cfg,'fixture','Check job 7100.',mock_job_ids=ids)
    assert not teacher.requests


def test_parallel_discovery_cannot_authorize_call_in_same_teacher_turn(cfg,monkeypatch):
    jid=fixture('RUNNING');executed=[];original=generate.execute
    def observed(name,args):
        executed.append(name);return original(name,args)
    monkeypatch.setattr(generate,'execute',observed)
    teacher=FakeTeacher([Completion('',[call('list_queue'),call(job_id=jid)]),Completion('Please identify the job.')])
    row=trace(cfg,teacher,'List my jobs and inspect one.',[jid])
    assert executed==['list_queue']
    assert not toolcheck.check_trace(row)['passed']


def test_guessed_accounting_cannot_launder_id_into_status(cfg,monkeypatch):
    executed=[]
    monkeypatch.setattr(generate,'execute',lambda *a:executed.append(a))
    teacher=FakeTeacher([Completion('',[call('job_accounting',job_id='8')]),
                         Completion('',[call(job_id='8')]),Completion('What is the job ID?')])
    row=trace(cfg,teacher,'Check my job; I requested 8 GPUs.',['8'])
    assert executed==[] and not toolcheck.check_trace(row)['passed']


def test_job_id_mentioned_only_in_log_text_is_not_target_evidence():
    from distillkit.job_status_guard import target_error
    c={'name':'job_status','arguments':{'job_id':'777'}}
    assert target_error(c,'Check my job.',[{'lines':['example job 777']}])
    assert target_error(c,'Check my job.',[{'job_id':'777','error':'not found'}])
    assert target_error(c,'Check my job.',[{'jobs':[{'job_id':'777'}]}]) is None
