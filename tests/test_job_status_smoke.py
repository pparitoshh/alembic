"""Exercise fixed-case orchestration with the actual OpenAI client over mock HTTP."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from distillkit.config import Config
from distillkit.teacher import Teacher
from distillkit.job_status_guard import LEGACY_POLICY_VERSION, POLICY_VERSION, RESOURCE_POLICY_VERSION
from distillkit.tools import CATALOG_VERSION, execute
from distillkit.io import read_jsonl
from test_pipeline import _cfg_dict


@pytest.mark.parametrize('version', [None, POLICY_VERSION, RESOURCE_POLICY_VERSION])
def test_smoke_orchestration_retains_all_cases_and_requests(tmp_path,monkeypatch,version):
    spec=importlib.util.spec_from_file_location('job_status_smoke',Path(__file__).parents[1]/'slurm/job_status_smoke.py')
    smoke=importlib.util.module_from_spec(spec);spec.loader.exec_module(smoke)
    d=_cfg_dict(tmp_path);d['generate']['gold_dir']=None;d['teacher']['api_key_env']=None
    if version is not None:d['generate']['job_status_policy_version']=version
    cfg=Config.model_validate(d)
    jid='8310001';status=execute('job_status',{'job_id':jid})
    cases=[{'id':'call','mode':'call','question':f'Check job {jid}.','expected_job_id':jid,'doc_id':'fixture'},
           {'id':'ask','mode':'ask','question':'Check my job.','expected_job_id':None,'doc_id':'fixture'},
           {'id':'none','mode':'none','question':'Explain status fields generally.','expected_job_id':None,'doc_id':'fixture'}]
    plan={'cases':cases,'mock_job_ids':[jid],'mock_catalog_version':CATALOG_VERSION}
    monkeypatch.setattr(smoke,'preflight',lambda root:(cfg,plan,'Synthetic fixture source.',{}))
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-cpu-test')
    def reply(request):
        body=json.loads(request.content);messages=body['messages'];q=messages[1]['content']
        if q==cases[0]['question'] and messages[-1]['role']!='tool':
            message={'role':'assistant','content':'Checking.','tool_calls':[{'id':'call_1','type':'function','function':{'name':'job_status','arguments':json.dumps({'job_id':jid})}}]};finish='tool_calls'
        else:
            text=f"The state is {status['state']}." if q==cases[0]['question'] else ('What is the job ID?' if q==cases[1]['question'] else 'Status reports observed job fields.')
            message={'role':'assistant','content':text};finish='stop'
        return httpx.Response(200,json={'id':'mock','object':'chat.completion','created':0,'model':cfg.teacher.model,'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}})
    class MockHTTPTeacher(Teacher):
        def __init__(self,config):
            super().__init__(config)
            self.client.close()
            self.client=OpenAI(api_key='synthetic-test-only',base_url='http://mock.invalid/v1',http_client=httpx.Client(transport=httpx.MockTransport(reply)))
    monkeypatch.setattr(smoke,'Teacher',MockHTTPTeacher)
    smoke.run(tmp_path,8000)
    rows=read_jsonl(tmp_path/'outputs/conversations.jsonl')
    requests=read_jsonl(tmp_path/'outputs/request_attempts.jsonl')
    assert len(rows)==3 and len(requests)==4
    assert all(r['status']=='complete' and r['verification']['passed'] for r in rows)
    assert all(r['deterministic_case_check']['exact_requested_call'] for r in rows)
    assert all(r['training_eligible'] is False for r in rows)
    assert all(r['tool_policy']['version']==(version or LEGACY_POLICY_VERSION) for r in rows)
    assert all(r['status']=='success' and r['messages'] for r in requests)
    assert {r['case_id'] for r in requests}=={'call','ask','none'}
