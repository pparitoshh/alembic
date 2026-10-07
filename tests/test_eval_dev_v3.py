"""Exposed development data/scoring regressions; no model or cluster requests."""
import copy
import hashlib
import json
from pathlib import Path
import unicodedata

import pytest

from distillkit import evaluate, gguf_eval
from distillkit.config import load_config
from distillkit.eval_tools_v2 import POLICY, POLICY_TEXT, PROTOCOL, system_prompt, answer_binding
from distillkit.io import read_jsonl
from distillkit.toolcheck import score_tool_item
from distillkit.tools import SCHEMAS, execute, mock_session

ROOT = Path(__file__).parents[1]
RELEASE = ROOT / 'data/eval/dev_v3'
TOOLS = read_jsonl(RELEASE / 'tools.jsonl')


def call(name, **arguments):
    return '<tool_call>' + json.dumps({'name':name, 'arguments':arguments}) + '</tool_call>'


def cfg(tmp_path):
    return load_config(ROOT / 'configs/qwen3_4b_qdora.yaml', [
        f'run_dir={tmp_path}', f'eval.file={RELEASE / "normal.jsonl"}',
        f'eval.tool_file={RELEASE / "tools.jsonl"}', f'eval.tool_protocol={PROTOCOL}'])


def test_three_reference_only_changes_have_exact_local_support_and_preserve_questions():
    old = read_jsonl(ROOT / 'data/eval/eval_all_v2.jsonl')
    new = read_jsonl(RELEASE / 'normal.jsonl')
    changes = json.loads((RELEASE / 'corrections.json').read_text())['changes']
    assert len(new) == len(old) == 66
    assert [a['id'] for a,b in zip(old,new) if a != b] == ['arr-003','req-002','req-011']
    assert [{k:v for k,v in r.items() if k != 'reference'} for r in old] == [
        {k:v for k,v in r.items() if k != 'reference'} for r in new]
    assert all(set(r)=={'id','doc_id','task','question','reference'} and r['reference'].strip() for r in new)
    for change in changes:
        source = (ROOT/change['source']['path']).read_bytes()
        assert hashlib.sha256(source).hexdigest() == change['source']['sha256']
        assert change['source']['quote'] in source.decode()
    assert 'directory must exist' not in new[2]['reference']
    req = {r['id']:r for r in new}
    assert 'preempt' not in req['req-002']['reference'] and 'node fails' not in req['req-002']['reference']
    assert '--mem' not in req['req-011']['reference'] and 'sacct -j <jobid> --format=JobID,State,ExitCode' in req['req-011']['reference']
    normal = lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
    assert len({normal(r['question']) for r in new})==66
    assert sum(r['id']=='req-016' for r in new)==1


@pytest.mark.parametrize('item', TOOLS, ids=lambda r:r['id'])
def test_reviewed_tool_labels_accept_the_supported_first_response(item):
    if item['expect']=='no_call':
        result=score_tool_item('Please identify the job.' if item['id'] in {'tool-011','tool-012','tool-013'} else 'A general answer.',item)
        assert result['first_response_pass'] and not result['response_quality_assessed']
    else:
        expected=item['expected_call']; expected=expected if isinstance(expected,list) else [expected]
        for option in expected:
            arguments={k:values[0] for k,values in option['arguments'].items() if values[0]!=''}
            result=score_tool_item(call(option['name'],**arguments),item)
            assert result['first_response_pass'] and result['valid'] and result['ast_ok'] and result['exec_ok']
    assert not result['task_completion_assessed']


@pytest.mark.parametrize('change',['case','command','not_test_only','missing_test_only','extra_argument'])
def test_script_wildcard_is_replaced_by_case_sensitive_exact_content(change):
    item=TOOLS[8]; script=item['question'].split('\n\n',1)[1]
    args={'script':script,'test_only':True}
    if change=='case':args['script']=script.replace('train.py','TRAIN.py')
    elif change=='command':args['script']=script.replace('train.py','another.py')
    elif change=='not_test_only':args['test_only']=False
    elif change=='missing_test_only':del args['test_only']
    else:args['partition']='boost_usr_prod'
    assert not score_tool_item(call('submit_job',**args),item)['first_response_pass']
    assert score_tool_item(call('submit_job',script=script.replace('\n','\r\n')+'\r\n',test_only=True),item)['first_response_pass']


@pytest.mark.parametrize('index',[10,11,12])
def test_missing_target_policy_rejects_discovery_guessing_and_actions(index):
    item=TOOLS[index]
    for text in [call('list_queue'),call('job_status',job_id='4718207'),call('cancel_job',job_id='4718207')]:
        assert not score_tool_item(text,item)['first_response_pass']


def test_allowed_readonly_discovery_is_not_a_false_call_and_cannot_hide_actions():
    # Synthetic explicitly permissive alternative; the shipped release uses clarify-first.
    item={**TOOLS[10], 'application_policy':'hpc-tools-readonly-discovery-v1','allow_lookup':True}
    score=score_tool_item(call('list_queue'),item)
    summary=evaluate.tool_summary([score],[item])
    assert score['allowed_discovery'] and score['first_response_pass']
    assert summary['tool_false_call_rate']==0 and summary['tool_raw_call_rate']==1
    for text in [call('list_queue',state='RUNNING'),call('gpu_availability'),call('submit_job',script='#!/bin/bash\ntrue'),call('list_queue')+call('cancel_job',job_id='99')]:
        assert not score_tool_item(text,item)['first_response_pass']
    parsed=score_tool_item('<tool_call>{"name":"list_queue","arguments":"{}"}</tool_call>',item)
    assert parsed['allowed_discovery']


def test_all_calls_and_malformed_delimiters_are_accounted_for():
    assert not score_tool_item(call('job_status',job_id='4718207')+call('cancel_job',job_id='4718207'),TOOLS[0])['first_response_pass']
    assert not score_tool_item('<tool_call>{"name":"list_queue"}',TOOLS[10])['first_response_pass']
    assert not score_tool_item('<tool_call>not json</tool_call>',TOOLS[10])['first_response_pass']
    assert not score_tool_item('  ',TOOLS[10])['first_response_pass']
    for name in [[],{},None,1]:
        assert not score_tool_item(call(name),TOOLS[0])['first_response_pass']


@pytest.mark.parametrize('arguments',[[],False,0,None,'','[]','null'])
def test_falsy_nonobject_arguments_never_become_empty_valid_arguments(arguments):
    discovery={**TOOLS[10],'application_policy':'hpc-tools-readonly-discovery-v1','allow_lookup':True}
    for item,name in [(TOOLS[9],'gpu_availability'),(discovery,'list_queue')]:
        text='<tool_call>'+json.dumps({'name':name,'arguments':arguments})+'</tool_call>'
        result=score_tool_item(text,item)
        assert not result['valid'] and not result['first_response_pass']


def test_evaluation_mock_actions_are_isolated():
    with mock_session():
        before=execute('job_status',{'job_id':'4718210'})
        assert score_tool_item(call('cancel_job',job_id='4718210'),TOOLS[3])['first_response_pass']
        assert execute('job_status',{'job_id':'4718210'})==before


def test_visible_policy_does_not_depend_on_hidden_labels_and_dataset_is_bound(tmp_path):
    c=cfg(tmp_path); normal,tools=evaluate._eval_rows(c)
    assert len(normal)==66 and len(tools)==16
    assert system_prompt(c,False)==c.student.system_prompt
    assert system_prompt(c,True)==c.student.system_prompt+'\n\n'+POLICY_TEXT
    assert all(t['application_policy']==POLICY and 'allow_lookup' not in t for t in tools)
    c.eval.tool_protocol='tool-first-call-v1'
    with pytest.raises(ValueError,match='protocol mismatch'):evaluate._eval_rows(c)


def test_permitted_discovery_has_a_visible_policy_and_separate_binding(tmp_path):
    from distillkit.eval_tools_v2 import DISCOVERY_POLICY
    c=cfg(tmp_path)
    before=answer_binding(c,['q'],[SCHEMAS])
    c.eval.tool_application_policy=DISCOVERY_POLICY
    prompt=system_prompt(c,True)
    assert DISCOVERY_POLICY in prompt and 'list_queue call with no arguments' in prompt
    assert 'at most ONE tool call' in prompt
    assert answer_binding(c,['q'],[SCHEMAS])!=before
    with pytest.raises(ValueError,match='visible application policy'):evaluate._eval_rows(c)
    rows=[{**TOOLS[10],'application_policy':DISCOVERY_POLICY,'allow_lookup':True}]
    c.eval.tool_file=tmp_path/'permissive.jsonl'
    c.eval.tool_file.write_text(json.dumps(rows[0])+'\n')
    assert evaluate._eval_rows(c)[1]==rows
    assert score_tool_item(call('list_queue'),rows[0])['first_response_pass']


def test_backend_generation_settings_invalidate_bindings(tmp_path):
    c=cfg(tmp_path)
    hf=answer_binding(c,['q'],[None]); gguf=answer_binding(c,['q'],[None],backend='gguf')
    c.train.fp16=True;c.train.bf16=False
    assert answer_binding(c,['q'],[None])!=hf
    c.export.num_ctx+=1
    assert answer_binding(c,['q'],[None],backend='gguf')!=gguf


def test_new_policy_never_reuses_legacy_answers_and_old_cache_is_preserved(tmp_path,monkeypatch):
    c=cfg(tmp_path);normal,tools=evaluate._eval_rows(c);questions=[r['question'] for r in normal+tools]
    old=json.dumps({'questions':questions,'adapter_mtime':None,'answers':{'base':['OLD']*82}})
    (tmp_path/'eval_answers.json').write_text(old)
    calls=[]
    def fake(cfg,questions,schemas):
        calls.append((questions,schemas));return {'base':['FAKE CPU FIXTURE']*len(questions)}
    monkeypatch.setattr(evaluate,'generate_answers',fake)
    assert evaluate.run_answers(c)['base'][0]=='FAKE CPU FIXTURE'
    assert len(calls)==1 and (tmp_path/'eval_answers.json').read_text()==old
    evaluate.run_answers(c);assert len(calls)==1
    c.student.system_prompt+=' Changed visible instruction.'
    evaluate.run_answers(c);assert len(calls)==2


def test_gguf_prompt_uses_same_policy_and_v3_cache_namespace(tmp_path,monkeypatch):
    from contextlib import contextmanager
    c=cfg(tmp_path);(tmp_path/'export').mkdir();(tmp_path/'export/model-Q4.gguf').write_text('synthetic')
    rendered=[]
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            rendered.append(messages);return json.dumps(messages)
    @contextmanager
    def server(*args,**kwargs):yield 'fake-no-network'
    monkeypatch.setattr(gguf_eval,'_tokenizer',lambda _:Tokenizer())
    monkeypatch.setattr(gguf_eval,'llama_server',server)
    monkeypatch.setattr(gguf_eval,'complete',lambda *args:'FAKE CPU FIXTURE')
    gguf_eval.answers(c,'Q4',['general','tool'],[None,SCHEMAS])
    assert rendered[0][0]['content']==c.student.system_prompt
    assert rendered[1][0]['content']==system_prompt(c,True)
    assert (tmp_path/'eval_answers_gguf_Q4_v3.json').exists()
    assert not (tmp_path/'eval_answers_gguf_Q4.json').exists()
