"""Production verification with fake support reviewers, never network inference."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from distillkit import generate, support, verify
from distillkit.config import Config
from distillkit.io import read_jsonl, write_jsonl
from distillkit.job_status_guard import JobStatusPolicy, WORKFLOW_VERSION
from distillkit.teacher import Completion
from distillkit.tools import CATALOG_VERSION, MOCK_VERSION, SCHEMAS
from test_job_status_workflow import FakeTeacher, call, fixture
from test_pipeline import _cfg_dict

SOURCE=(Path(__file__).parents[1]/'docs/job_status_contract.md').read_text()
OBSERVED='A job status and nonzero exit code do not explain the cause of an application failure because they only indicate that the job ended abnormally. They do not reveal whether the issue was due to resource limits, software errors, runtime exceptions, or other factors. For example, a job might fail due to an out-of-memory condition, but the exit code alone does not distinguish this from a bug in the application code. To diagnose the root cause, additional information such as log files or accounting data is necessary.'


@pytest.fixture
def cfg(tmp_path):
    d=_cfg_dict(tmp_path);d['generate'].update(gold_dir=None,job_status_discovery='clarify_first')
    d['seeds']['dir']=str(tmp_path/'seeds')
    d['verify'].update(require_grounding_review=True,grounding_reviews=str(tmp_path/'support.jsonl'))
    c=Config.model_validate(d)
    c.seeds.dir.mkdir()
    (c.seeds.dir/'status_test.md').write_text(SOURCE)
    return c


def record(cfg, answer, source=SOURCE, responses=None):
    if source!=SOURCE:
        (cfg.seeds.dir/'status_test.md').write_text(source)
    jid=fixture('FAILED')
    question='Explain whether the status result identifies an application failure cause.'
    if responses:
        question=f'Check job {jid} and report only what is established.'
    policy,audit=JobStatusPolicy(),{}
    transcript=generate.tool_trace(FakeTeacher(responses or [Completion(answer)]),cfg,source,question,
                                   mock_job_ids=[jid],tool_policy=policy,audit=audit)
    return {'id':'support-fixture','doc_id':'status_test','source_kind':'full',
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'question':question,
            'mode':'call' if responses else 'none','messages':transcript,'tools':SCHEMAS,
            'mock_version':MOCK_VERSION,'mock_catalog_version':CATALOG_VERSION,'mock_job_ids':[jid],
            'tool_policy':policy.as_dict(),'runtime_audit':audit,'workflow_version':WORKFLOW_VERSION}


class FakeReviewer:
    model='fake-reviewer-no-inference'
    def __init__(self,review):
        self.review=review;self.requests=[]
    def complete(self,messages,**kwargs):
        self.requests.append((messages,kwargs))
        return Completion(json.dumps(self.review))


def report(row, source=SOURCE, verdict='supported', kind='assertion', evidence=None):
    packet=support.evidence_packet(row,source)
    segments=[{'turn':t['turn'],'text':t['text'],'kind':kind,'support':verdict,
               'evidence':evidence if evidence is not None else ([{'evidence_id':'source','quote':'A failed state and exit code alone do not identify a cause.'}] if verdict=='supported' else []),
               'explanation':'Synthetic reviewer fixture: inspect cited support and preserve uncertainty.'} for t in packet['turns']]
    reviewer=FakeReviewer({'segments':segments,'uncertainty':''})
    result=support.request_review(reviewer,row,source)
    assert len(reviewer.requests)==1
    return result


def validate_run(cfg,row,reviews):
    write_jsonl(cfg.run_dir/'generated.jsonl',[row]);write_jsonl(cfg.verify.grounding_reviews,reviews)
    verify.run(cfg)
    return [read_jsonl(cfg.run_dir/name) for name in ['verified.jsonl','rejected.jsonl','pending_review.jsonl']]


def test_observed_hypothetical_causes_rejected_by_real_verification(cfg):
    row=record(cfg,OBSERVED)
    review=report(row,verdict='unsupported',kind='hypothetical')
    review['review']['segments'][0]['explanation']='Resource/software/runtime possibilities and OOM versus application bug are absent from the source; this is not an asserted diagnosis of an actual job.'
    review['raw_response']=json.dumps(review['review'])
    kept,rejected,pending=validate_run(cfg,row,[review])
    assert not kept and not pending and rejected[0]['reject_reason']=='unsupported_claim'
    assert rejected[0]['grounding_checks']['segments'][0]['kind']=='hypothetical'


def test_source_supported_paraphrase_passes_without_verbatim_answer(cfg):
    row=record(cfg,'Those status fields do not identify the cause of failure.')
    kept,rejected,pending=validate_run(cfg,row,[report(row)])
    assert len(kept)==1 and not rejected and not pending


def test_supported_hypothetical_example_is_not_blanket_rejected(cfg):
    source=SOURCE+'\nFor example, two jobs with identical names can have different GPU counts.\n'
    row=record(cfg,'For example, suppose two jobs share a name but have different GPU counts.',source)
    review=report(row,source,kind='hypothetical',evidence=[{'evidence_id':'source','quote':'two jobs with identical names can have different GPU counts'}])
    kept,rejected,pending=validate_run(cfg,row,[review])
    assert len(kept)==1 and not rejected and not pending


@pytest.mark.parametrize('failure',['missing','uncertain','stale','malformed','duplicate','bad_quote','omitted_claim','wrong_protocol','altered_raw','qualification','bad_reviewer','bad_time','naive_time'])
def test_missing_or_unreliable_review_is_pending_not_accepted_or_rejected(cfg,failure):
    row=record(cfg,'Status does not identify a cause. A log is a separate observation.')
    r=report(row);reports=[r]
    segment=r['review']['segments'][0]
    if failure=='missing':reports=[]
    elif failure=='uncertain':segment['support']='uncertain'
    elif failure=='stale':r['packet_sha256']='stale'
    elif failure=='malformed':r['review']={}
    elif failure=='duplicate':reports=[r,copy.deepcopy(r)]
    elif failure=='bad_quote':segment['evidence'][0]['quote']='invented source sentence'
    elif failure=='omitted_claim':segment['text']='Status does not identify a cause.'
    elif failure=='wrong_protocol':r['protocol']='unknown'
    elif failure=='qualification':r['review']['uncertainty']='I cannot establish support for the second sentence.'
    elif failure=='bad_reviewer':r['reviewer']={'invented':'metadata'}
    elif failure=='bad_time':r['created_utc']=True
    elif failure=='naive_time':r['created_utc']='2026-01-01T00:00:00'
    if failure!='altered_raw':r['raw_response']=json.dumps(r['review'])
    else:r['raw_response']='{}'
    kept,rejected,pending=validate_run(cfg,row,reports)
    assert not kept and not rejected and len(pending)==1


def test_later_result_cannot_support_an_earlier_assistant_guess(cfg):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('The job has failed.',[call(job_id=jid)]),Completion('Its state is FAILED.')])
    r=report(row,evidence=[{'evidence_id':'tool:2','quote':'FAILED'}])
    kept,rejected,pending=validate_run(cfg,row,[r])
    assert not kept and not rejected and len(pending)==1
    assert 'unavailable before' in pending[0]['grounding_checks']['reason']


def test_supported_positive_call_and_earlier_intent_have_distinct_support(cfg):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('I will inspect its status.',[call(job_id=jid)]),Completion('The state is FAILED.')])
    r=report(row)
    first,final=r['review']['segments']
    first.update(kind='question_or_intent',support='not_factual',evidence=[])
    final['evidence']=[{'evidence_id':'tool:2','quote':'FAILED'}]
    r['raw_response']=json.dumps(r['review'])
    kept,rejected,pending=validate_run(cfg,row,[r])
    assert len(kept)==1 and not rejected and not pending


def test_asserted_unsupported_diagnosis_is_distinct_from_hypothetical(cfg):
    row=record(cfg,'Your job failed because the program ran out of memory.')
    r=report(row,verdict='unsupported',kind='assertion')
    kept,rejected,pending=validate_run(cfg,row,[r])
    assert not kept and not pending
    assert rejected[0]['grounding_checks']['segments'][0]['kind']=='assertion'


def test_unsupported_command_remains_a_hard_failure_without_a_support_review(cfg):
    row=record(cfg,'Read stderr with:\n```bash\nsacct -j 7199999 --format=JobID,State,ExitCode,Reason\n```')
    kept,rejected,pending=validate_run(cfg,row,[])
    assert not kept and not pending and rejected[0]['reject_reason']=='workflow_unsupported_response'


def test_record_schema_policy_source_or_transcript_changes_invalidate_review(cfg):
    row=record(cfg,'Status does not identify the failure cause.')
    r=report(row)
    for field,value in [('tool_policy',JobStatusPolicy('allow_readonly').as_dict()),
                        ('tools',[]),('source_sha256','changed')]:
        changed=copy.deepcopy(row);changed[field]=value
        kept,rejected,pending=validate_run(cfg,changed,[r])
        assert not kept and (rejected or pending)
    changed=copy.deepcopy(row);changed['messages'][-1]['content']='A different answer.'
    assert support.check_support(changed,SOURCE,r)['status']=='uncertain'


def test_hidden_gold_and_expected_fields_are_not_support_evidence(cfg):
    row=record(cfg,'Status does not identify the failure cause.')
    before=support.evidence_packet(row,SOURCE)
    row.update(reference='Invented cause.',expected_mode='none',expected_job_id='99111',gold_verdict='accept')
    assert support.evidence_packet(row,SOURCE)==before


def test_explicit_user_scenario_values_can_be_cited_without_becoming_live_facts(cfg):
    row=record(cfg,'Under your stated assumption, the requested duration is two hours.')
    row['messages'][0]['content']='Assume I request a two-hour duration; explain my stated requirement.'
    row['question']=row['messages'][0]['content']
    r=report(row,evidence=[{'evidence_id':'user:0','quote':'Assume I request a two-hour duration'}])
    assert support.check_support(row,SOURCE,r)['status']=='supported'


def test_malformed_review_file_does_not_abort_or_accept_records(cfg):
    row=record(cfg,'Status alone does not identify a cause.')
    write_jsonl(cfg.run_dir/'generated.jsonl',[row])
    cfg.verify.grounding_reviews.write_text('{broken\n')
    verify.run(cfg)
    assert not read_jsonl(cfg.run_dir/'verified.jsonl')
    assert len(read_jsonl(cfg.run_dir/'pending_review.jsonl'))==1


@pytest.mark.parametrize('mode',['prose',None,'misspelled'])
def test_mislabeled_tool_record_cannot_bypass_replay(cfg,mode):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('',[call(job_id=jid)]),Completion('Its state is RUNNING.')])
    row['messages'][2]['content']=json.dumps({'job_id':jid,'state':'RUNNING'})
    row['mode']=mode
    r=report(row,evidence=[{'evidence_id':'tool:2','quote':'RUNNING'}])
    kept,rejected,pending=validate_run(cfg,row,[r])
    assert not kept and not pending and rejected[0]['reject_reason']=='tool_mode_mismatch'


def test_review_sees_executed_call_identity_only_after_its_result(cfg):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('I will check status.',[call(job_id=jid)]),Completion('The state is FAILED.')])
    before,after=support.evidence_packet(row,SOURCE)['turns']
    assert not any(k.startswith('executed_call:') for k in before['evidence'])
    evidence=json.loads(after['evidence']['executed_call:2'])
    assert evidence['call']=={'name':'job_status','arguments':{'job_id':jid}}
    assert evidence['result']['state']=='FAILED'


@pytest.mark.parametrize('content',[None,[]])
def test_empty_call_preamble_or_malformed_content_cannot_abort_verification(cfg,content):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('',[call(job_id=jid)]),Completion('Its state is FAILED.')])
    row['messages'][1]['content']=content
    if content is None:
        reports=[report(row,evidence=[{'evidence_id':'tool:2','quote':'FAILED'}])]
    else:
        reports=[]
    good=record(cfg,'A failed state and exit code alone do not identify a cause.')
    good['id']='second-valid-row'
    good['question']=good['messages'][0]['content']='Generally, what is the limit of these fields as evidence?'
    write_jsonl(cfg.run_dir/'generated.jsonl',[row,good])
    write_jsonl(cfg.verify.grounding_reviews,[*reports,report(good)])
    verify.run(cfg)
    kept=read_jsonl(cfg.run_dir/'verified.jsonl');pending=read_jsonl(cfg.run_dir/'pending_review.jsonl')
    assert any(r['id']==good['id'] for r in kept)
    assert len(kept)==(2 if content is None else 1)
    assert len(pending)==(0 if content is None else 1)


def test_pending_question_does_not_displace_a_supported_near_duplicate(cfg):
    pending=record(cfg,'Status alone does not identify a cause.');pending['id']='a'
    supported=copy.deepcopy(pending);supported['id']='b'
    pending['question']=pending['messages'][0]['content']='Explain the exact limits of job status and exit code evidence about an application failure cause.'
    supported['question']=supported['messages'][0]['content']=pending['question']+' Please.'
    write_jsonl(cfg.run_dir/'generated.jsonl',[pending,supported])
    write_jsonl(cfg.verify.grounding_reviews,[report(supported)])
    verify.run(cfg)
    assert [r['id'] for r in read_jsonl(cfg.run_dir/'verified.jsonl')]==['b']
    assert [r['id'] for r in read_jsonl(cfg.run_dir/'pending_review.jsonl')]==['a']
    assert read_jsonl(cfg.run_dir/'rejected.jsonl')==[]


def test_action_argument_content_is_not_approved_by_text_only_protocol(cfg):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('',[call(job_id=jid)]),Completion('The state is FAILED.')])
    row['messages'][1]['tool_calls'][0]['function']={'name':'submit_job','arguments':{'script':'synthetic script'}}
    r=report(row)
    assert support.check_support(row,SOURCE,r)['status']=='uncertain'
    assert 'separate content review' in support.check_support(row,SOURCE,r)['reason']
