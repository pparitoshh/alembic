"""CPU-only exposed development regressions. All V2 reviews are synthetic fixtures.

They prove contract enforcement, not that the real reviewer has improved. The original
ten conversations and V1 reviews are preserved; none are training records.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from distillkit import support, support_v2 as v2, verify
from distillkit.io import read_jsonl, write_jsonl
from distillkit.teacher import Completion
from test_job_status_workflow import call, fixture
from test_support_review import cfg, record, SOURCE, FakeReviewer, validate_run

CAPTURED=Path(__file__).parent/'fixtures/support_checkpoint06'
ROWS=read_jsonl(CAPTURED/'records.jsonl')
SOURCE_CAPTURED=(CAPTURED/'source.md').read_text()
REVIEWS=read_jsonl(CAPTURED/'reviews_v1.jsonl')


def test_captured_diagnostics_cannot_become_training_examples():
    from distillkit.records import training_example
    for row in ROWS:
        assert row['purpose']=='diagnostic_only' and row['training_eligible'] is False
        with pytest.raises(ValueError,match='diagnostic'):
            training_example(row, 'Synthetic test system prompt.')


def entry(packet, origin):
    return next(k for k,v in packet['registry'].items() if v['origin']==origin)


def payload(row, source=SOURCE):
    packet=v2.evidence_packet(row,source)
    return {'segments':[{'turn':t['turn'],'text':t['text'],'kind':'assertion',
        'support':'supported','evidence_ids':[entry(packet,'source')],
        'explanation':'Synthetic test judgment; source supports the stated limitation.'}
        for t in packet['turns']], 'uncertainty':{'present':False,'reason':None}}


def request(row, review, source=SOURCE):
    fake=FakeReviewer(review)
    report=support.request_review(fake,row,source,protocol=v2.PROTOCOL)
    assert len(fake.requests)==1
    prompt,kwargs=fake.requests[0]
    assert kwargs['temperature']==0 and kwargs['response_format']['json_schema']['name']=='SupportReviewV2'
    assert json.loads(prompt[1]['content'])==v2.evidence_packet(row,source)
    return report


def run_v2(cfg,row,review,source=SOURCE):
    cfg.verify.grounding_review_protocol=v2.PROTOCOL
    (cfg.seeds.dir/(row['doc_id']+'.md')).write_text(source)
    return validate_run(cfg,row,[request(row,review,source)])


def test_preserved_captured_reviews_reproduce_ten_pending_with_actual_verify(cfg):
    (cfg.seeds.dir/'distillkit_job_status.md').write_text(SOURCE_CAPTURED)
    write_jsonl(cfg.run_dir/'generated.jsonl',ROWS)
    write_jsonl(cfg.verify.grounding_reviews,REVIEWS)
    verify.run(cfg)
    assert read_jsonl(cfg.run_dir/'verified.jsonl')==[]
    assert read_jsonl(cfg.run_dir/'rejected.jsonl')==[]
    pending=read_jsonl(cfg.run_dir/'pending_review.jsonl')
    assert len(pending)==10
    reasons=[r['grounding_checks'].get('reason') for r in pending]
    assert reasons.count('citation absent, altered or unavailable before this claim')==7
    assert reasons.count('unreviewed assistant text')==1
    assert sum(r['grounding_checks'].get('uncertainty')=='none' for r in pending)==2
    manifest=json.loads((CAPTURED/'manifest.json').read_text())
    for name,sha in manifest['fixture_hashes'].items():
        assert hashlib.sha256((CAPTURED/name).read_bytes()).hexdigest()==sha


def test_supported_source_claim_uses_registry_not_reconstructed_quotes(cfg):
    row=record(cfg,'Those status fields alone cannot establish the cause of failure.')
    kept,rejected,pending=run_v2(cfg,row,payload(row))
    assert len(kept)==1 and not rejected and not pending
    check=kept[0]['grounding_checks']
    assert check['uncertainty']=={'present':False,'reason':None}
    assert 'fallible' in check['limitation']


def test_previous_tool_result_and_future_intent_are_distinct(cfg):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('I will check its status.',[call(job_id=jid)]),
                                Completion('The returned state is FAILED.')])
    packet=v2.evidence_packet(row,SOURCE); data=payload(row)
    first,last=data['segments']
    first.update(kind='question_or_intent',support='not_factual',evidence_ids=[])
    last['evidence_ids']=[entry(packet,'tool:2'),entry(packet,'executed_call:2')]
    assert all(e not in packet['turns'][0]['available_evidence_ids'] for e in last['evidence_ids'])
    kept,rejected,pending=run_v2(cfg,row,data)
    assert len(kept)==1 and not rejected and not pending


@pytest.mark.parametrize('fault',['altered','missing','answer_as_source','future','uncovered_addition',
                                'string_none','true_uncertainty','false_with_text','registry_hash','raw_changed'])
def test_invalid_or_uncertain_reviews_never_approve(cfg,fault):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('I will check its status.',[call(job_id=jid)]),
         Completion('The returned state is FAILED. A special imaginary condition caused it.')])
    packet=v2.evidence_packet(row,SOURCE); data=payload(row)
    data['segments'][0].update(kind='question_or_intent',support='not_factual',evidence_ids=[])
    data['segments'][1]['evidence_ids']=[entry(packet,'tool:2')]
    if fault=='altered':data['segments'][1]['evidence_ids'][0]+='changed'
    elif fault=='missing':data['segments'][1]['evidence_ids']=[]
    elif fault=='answer_as_source':data['segments'][1]['evidence_ids']=['assistant:3']
    elif fault=='future':data['segments'][0]['evidence_ids']=[entry(packet,'tool:2')]
    elif fault=='uncovered_addition':data['segments'][1]['text']='The returned state is FAILED.'
    elif fault=='string_none':data['uncertainty']='none'
    elif fault=='true_uncertainty':data['uncertainty']={'present':True,'reason':'The cause has no evidence.'}
    elif fault=='false_with_text':data['uncertainty']={'present':False,'reason':'none'}
    r=request(row,data)
    if fault=='registry_hash':r['registry_sha256']='changed'
    elif fault=='raw_changed':r['raw_response']='{}'
    cfg.verify.grounding_review_protocol=v2.PROTOCOL
    kept,rejected,pending=validate_run(cfg,row,[r])
    assert not kept and not rejected and len(pending)==1


def test_unexecuted_or_altered_result_is_rejected_before_reviewer_request(cfg):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('',[call(job_id=jid)]),Completion('Its state is FAILED.')])
    row['messages'][2]['content']='{"state":"MADE_UP"}'
    fake=FakeReviewer({})
    with pytest.raises(ValueError,match='mock replay'):
        support.request_review(fake,row,SOURCE,protocol=v2.PROTOCOL)
    assert fake.requests==[]


@pytest.mark.parametrize('name,args',[('submit_job',{'script':'#!/bin/bash\ntrue','test_only':True}),
                                    ('cancel_job',{'job_id':'4718210'})])
def test_out_of_scope_action_never_enters_registry_or_reviewer_request(cfg,name,args):
    row=record(cfg,'Synthetic answer.')
    row.pop('tool_policy');row.pop('workflow_version')
    row['messages'][1]['tool_calls']=[{'type':'function','function':{'name':name,'arguments':args}}]
    fake=FakeReviewer({})
    with pytest.raises(ValueError,match='read-only diagnostics'):
        support.request_review(fake,row,SOURCE,protocol=v2.PROTOCOL)
    assert fake.requests==[]


@pytest.mark.parametrize('fault',['omit_turn','reorder','repeat','source','schema','policy','prompt'])
def test_coverage_and_all_evidence_bindings_are_checked(cfg,fault):
    jid=fixture('FAILED')
    row=record(cfg,'',responses=[Completion('I will inspect status.',[call(job_id=jid)]),Completion('Its state is FAILED.')])
    data=payload(row);packet=v2.evidence_packet(row,SOURCE)
    data['segments'][0].update(kind='question_or_intent',support='not_factual',evidence_ids=[])
    data['segments'][1]['evidence_ids']=[entry(packet,'tool:2')]
    if fault=='omit_turn':data['segments'].pop(0)
    elif fault=='reorder':data['segments'].reverse()
    elif fault=='repeat':data['segments'].append(copy.deepcopy(data['segments'][1]))
    r=request(row,data);source=SOURCE
    if fault=='source':source+='\nAltered source.'
    elif fault=='schema':row['tools']=row['tools'][:-1]
    elif fault=='policy':row['tool_policy']['discovery']='allow_readonly'
    elif fault=='prompt':r['prompt_sha256']='changed'
    assert support.check_support(row,source,r,protocol=v2.PROTOCOL)['status']=='uncertain'


def test_blocked_call_result_and_answer_text_are_not_evidence(cfg):
    row=record(cfg,'',responses=[Completion('',[call(job_id='99999999')])])
    # The visible question identifies another fixture; production guard blocks this ID.
    packet=v2.evidence_packet(row,SOURCE)
    assert all(not e['origin'].startswith(('assistant:', 'tool:', 'executed_call:'))
               for e in packet['registry'].values())
    row=record(cfg,'ANSWER_ONLY_UNIQUE_SENTENCE.')
    assert all('ANSWER_ONLY_UNIQUE_SENTENCE' not in e['content'] for e in v2.evidence_packet(row,SOURCE)['registry'].values())


@pytest.mark.parametrize('cid',['js4-supported-hypothesis','js4-suspicion-not-evidence'])
def test_captured_unsupported_addenda_are_rejected_by_synthetic_evidence_review(cfg,cid):
    row=next(r for r in ROWS if r['id']==cid)
    data=payload(row,SOURCE_CAPTURED)
    data['segments'][0].update(kind='mixed',support='unsupported',evidence_ids=[],
        explanation=('The allocation field does not establish configuration/scheduling history.' if cid.endswith('hypothesis')
                     else 'A cause taxonomy and memory-diagnostic rule are absent from the supplied evidence.'))
    kept,rejected,pending=run_v2(cfg,row,data,SOURCE_CAPTURED)
    assert not kept and not pending and rejected[0]['reject_reason']=='unsupported_claim'


def test_supported_hypothesis_is_allowed_and_coverage_still_checks_additions(cfg):
    row=record(cfg,'For example, a job named train_llm can have a zero gpus field; its name alone does not show GPU use.')
    data=payload(row);data['segments'][0]['kind']='hypothetical'
    assert len(run_v2(cfg,row,data)[0])==1
    row=next(r for r in ROWS if r['id']=='js4-supported-hypothesis')
    data=payload(row,SOURCE_CAPTURED)
    data['segments'][0]['text']=data['segments'][0]['text'].split(' This means')[0]
    kept,rejected,pending=run_v2(cfg,row,data,SOURCE_CAPTURED)
    assert not kept and not rejected and pending[0]['grounding_checks']['reason']=='unreviewed assistant text'


def test_captured_intent_false_positive_has_a_valid_synthetic_review(cfg):
    row=next(r for r in ROWS if r['id']=='js4-select-sole')
    packet=v2.evidence_packet(row,SOURCE_CAPTURED); data=payload(row,SOURCE_CAPTURED)
    data['segments'][0].update(kind='question_or_intent',support='not_factual',evidence_ids=[])
    data['segments'][1].update(kind='mixed',evidence_ids=[entry(packet,'tool:2')],
                               explanation='Observed sole result followed by future intent, not an executed-action claim.')
    data['segments'][2]['evidence_ids']=[entry(packet,'tool:4')]
    assert len(run_v2(cfg,row,data,SOURCE_CAPTURED)[0])==1


def test_valid_id_does_not_prove_entailment_or_override_coordinator_review(cfg):
    row=record(cfg,'An imaginary unsupported condition caused your job failure.')
    # Deliberately WRONG synthetic reviewer verdict, with a real but irrelevant ID.
    r=request(row,payload(row))
    result=support.check_support(row,SOURCE,r,protocol=v2.PROTOCOL)
    assert result['status']=='supported'
    assert 'fallible' in result['limitation']
    # This limitation is explicit: deterministic citation checks are not semantic proof.


def test_v1_is_default_and_never_translated_to_v2(cfg):
    row=ROWS[0];r=REVIEWS[0]
    assert cfg.verify.grounding_review_protocol==support.PROTOCOL
    assert support.check_support(row,SOURCE_CAPTURED,r)['status']=='uncertain'
    assert support.check_support(row,SOURCE_CAPTURED,r,protocol=v2.PROTOCOL)['reason']=='stale or mismatched review binding'


def test_source_policy_transcript_binding_and_hidden_fields(cfg):
    row=record(cfg,'The fields alone do not identify a cause.')
    packet=v2.evidence_packet(row,SOURCE)
    changed=copy.deepcopy(row);changed.update(expected_mode='bad',reference='private answer',gold_verdict='accept')
    assert v2.evidence_packet(changed,SOURCE)==packet
    r=request(row,payload(row))
    changed=copy.deepcopy(row);changed['messages'][-1]['content']+=' Extra assertion.'
    assert support.check_support(changed,SOURCE,r,protocol=v2.PROTOCOL)['status']=='uncertain'
    row['doc_id']='slurm_job_arrays';cfg.seeds.eval_docs=['slurm_job_arrays']
    cfg.verify.grounding_review_protocol=v2.PROTOCOL
    kept,rejected,pending=validate_run(cfg,row,[r])
    assert not kept and not pending and rejected[0]['reject_reason']=='heldout_source'
