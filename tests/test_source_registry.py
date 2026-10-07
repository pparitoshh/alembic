"""CPU regression fixtures; production paths with fake teachers, no inference."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillkit import generate, support, support_v2, verify, train
from distillkit.config import Config
from distillkit.io import read_jsonl, write_jsonl
from distillkit.records import training_example
from distillkit.seeds import load_chunks
from distillkit.source_registry import admitted_sources
from distillkit.teacher import Completion
from test_pipeline import _cfg_dict

TEXT='The synthetic scanner reports a count. The count alone does not establish the cause of a failure.'


@pytest.fixture
def setup(tmp_path):
    d=_cfg_dict(tmp_path);d['seeds'].update(dir=str(tmp_path/'seeds'),registry=str(tmp_path/'registry.json'),eval_docs=['held'],chunk_chars=8000)
    d['generate'].update(gold_dir=None,questions_per_chunk=1,answers_per_question=1,personas=['a test user'],task_types=['concept'],tool_fraction=0)
    d['verify'].update(require_grounding_review=True,grounding_review_protocol='source-support-v2',grounding_reviews=str(tmp_path/'review.jsonl'))
    cfg=Config.model_validate(d);cfg.seeds.dir.mkdir();cfg.run_dir.mkdir()
    (cfg.seeds.dir/'fixture.md').write_text(TEXT)
    document={'doc_id':'fixture','path':'fixture.md','sha256':hashlib.sha256(TEXT.encode()).hexdigest(),'family':'original',
        'approved_for_training':True,'origin':'synthetic regression fixture','source_version':'fixture-v1',
        'permission_evidence':'synthetic fixture only','admission_review_sha256':'a'*64}
    registry={'version':'source-family-registry-v1','families':{'original':{'split':'train','parents':[],'lineage_reviewed':True}},'documents':[document]}
    cfg.seeds.registry.write_text(json.dumps(registry))
    return cfg,registry


def save(cfg,registry):cfg.seeds.registry.write_text(json.dumps(registry))


@pytest.mark.parametrize('fault',['development','final','unknown_parent','unreviewed','cycle','transitive_final',
                                  'duplicate_id','duplicate_path','unlisted','changed_bytes','not_admitted','escape'])
def test_forbidden_or_unresolved_sources_stop_before_teacher(setup,monkeypatch,fault):
    cfg,r=setup
    if fault in ('development','final'):r['families']['original']['split']=fault
    elif fault=='unknown_parent':r['families']['original']['parents']=['unknown']
    elif fault=='unreviewed':r['families']['original']['lineage_reviewed']=False
    elif fault=='cycle':r['families']['original']['parents']=['original']
    elif fault=='transitive_final':
        r['families']['original']['parents']=['copy_alias'];r['families']['copy_alias']={'split':'train','parents':['protected'],'lineage_reviewed':True}
        r['families']['protected']={'split':'final','parents':[],'lineage_reviewed':True}
    elif fault=='duplicate_id':r['documents'].append(copy.deepcopy(r['documents'][0]))
    elif fault=='duplicate_path':r['documents'].append({**r['documents'][0],'doc_id':'alias'})
    elif fault=='unlisted':(cfg.seeds.dir/'extra.md').write_text('Unlisted source.')
    elif fault=='changed_bytes':(cfg.seeds.dir/'fixture.md').write_text('Changed source.')
    elif fault=='not_admitted':r['documents'][0]['approved_for_training']=False
    elif fault=='escape':r['documents'][0]['path']='../fixture.md'
    save(cfg,r);constructed=[]
    def forbidden(*args,**kw):constructed.append(True);raise AssertionError('teacher must not be constructed')
    monkeypatch.setattr(generate,'Teacher',forbidden)
    with pytest.raises(ValueError):generate.run(cfg)
    assert not constructed and not (cfg.run_dir/'questions.jsonl').exists()


def test_explicit_eval_docs_still_excluded(setup):
    cfg,r=setup
    held_text='A different synthetic source reserved for evaluation.'
    (cfg.seeds.dir/'held.md').write_text(held_text)
    r['documents'].append({**r['documents'][0],'doc_id':'held','path':'held.md','family':'eval','approved_for_training':False,'sha256':hashlib.sha256(held_text.encode()).hexdigest()})
    r['families']['eval']={'split':'final','parents':[],'lineage_reviewed':True};save(cfg,r)
    train,held=load_chunks(cfg)
    assert {c['doc_id'] for c in train}=={'fixture'} and {c['doc_id'] for c in held}=={'held'}
    assert all(c['source_family']=='original' for c in train)


@pytest.mark.parametrize('fault',['same_family','derived_family','same_bytes'])
def test_explicit_holdout_reconciled_with_family_and_exact_content(setup,monkeypatch,fault):
    cfg,r=setup;held_text=TEXT if fault=='same_bytes' else 'Unique held-out facts.'
    (cfg.seeds.dir/'held.md').write_text(held_text)
    r['families']['declared_eval']={'split':'train','parents':[],'lineage_reviewed':True}
    family='original' if fault=='same_family' else 'declared_eval'
    if fault=='derived_family':r['families']['original']['parents']=['declared_eval']
    r['documents'].append({**r['documents'][0],'doc_id':'held','path':'held.md','family':family,'sha256':hashlib.sha256(held_text.encode()).hexdigest()})
    save(cfg,r);constructed=[]
    def forbidden(*args,**kwargs):constructed.append(True);raise AssertionError('teacher must not be constructed')
    monkeypatch.setattr(generate,'Teacher',forbidden)
    with pytest.raises(ValueError,match='explicit eval_docs'):generate.run(cfg)
    assert not constructed


def test_gold_entry_cannot_override_family_assignment(setup):
    cfg,r=setup;gold=cfg.run_dir/'gold';gold.mkdir();cfg.generate.gold_dir=gold
    raw=json.dumps({'messages':[{'role':'user','content':'Example?'},{'role':'assistant','content':'Example.'}]}).encode()
    (gold/'prose.json').write_bytes(raw)
    (gold/'provenance.json').write_text(json.dumps({'schema_version':1,'examples':{'prose.json':{'sha256':hashlib.sha256(raw).hexdigest(),'sources':[{'doc_id':'fixture','split':'train','sha256':hashlib.sha256(TEXT.encode()).hexdigest()}]}}}))
    assert '<example>' in generate._gold(cfg,'prose')
    r['families']['original']['split']='development';save(cfg,r)
    with pytest.raises(ValueError,match='resolves to development'):generate._gold(cfg,'prose')


def test_separately_stored_final_family_cannot_share_train_ancestor(setup,monkeypatch):
    cfg,r=setup
    r['families']['remote_final']={'split':'final','parents':['original'],'lineage_reviewed':True}
    save(cfg,r);constructed=[]
    def forbidden(*args,**kwargs):constructed.append(True);raise AssertionError('teacher must not be constructed')
    monkeypatch.setattr(generate,'Teacher',forbidden)
    with pytest.raises(ValueError,match='shares held-out'):generate.run(cfg)
    assert not constructed


class FakeTeacher:
    model='synthetic-no-model-run'
    requests=[]
    def __init__(self,cfg):self.cfg=cfg
    def map(self,fn,items):return [fn(x) for x in items]
    def chat_json(self,system,user,schema):
        self.requests.append(('question',user));return SimpleNamespace(question='Does the synthetic scanner count establish the cause of failure?'),'synthetic'
    def complete(self,messages,**kwargs):
        self.requests.append(('answer',messages));return Completion('The count alone does not establish the cause of a failure.')


def test_production_prose_generation_review_verify_loader_and_stale_binding(setup,monkeypatch):
    cfg,r=setup;FakeTeacher.requests=[];monkeypatch.setattr(generate,'Teacher',FakeTeacher)
    generate.run(cfg);rows=read_jsonl(cfg.run_dir/'generated.jsonl');assert len(rows)==1
    row=rows[0];assert len(FakeTeacher.requests)==2 and row['source_family']=='original'
    assert TEXT in FakeTeacher.requests[0][1]
    source=support.source_for_record(cfg,row);packet=support_v2.evidence_packet(row,source)
    eid=next(k for k,v in packet['registry'].items() if v['origin']=='source')
    payload={'segments':[{'turn':t['turn'],'text':t['text'],'kind':'assertion','support':'supported','evidence_ids':[eid],'explanation':'Synthetic fixture: exactly supported by the synthetic source.'} for t in packet['turns']], 'uncertainty':{'present':False,'reason':None}}
    class Reviewer:
        model='fake-reviewer-not-inference'
        def complete(self,messages,**kwargs):
            assert json.loads(messages[1]['content'])==packet
            return Completion(json.dumps(payload))
    report=support.request_review(Reviewer(),row,source,protocol='source-support-v2')
    write_jsonl(cfg.verify.grounding_reviews,[report]);verify.run(cfg)
    kept=read_jsonl(cfg.run_dir/'verified.jsonl');assert len(kept)==1
    example=training_example(kept[0],cfg.student.system_prompt)
    assert example['messages'][-1]['content']==row['messages'][-1]['content']
    assert not read_jsonl(cfg.run_dir/'pending_review.jsonl')
    # No regeneration is allowed to launder an earlier registry binding.
    r['documents'][0]['source_version']='fixture-v2';save(cfg,r)
    with pytest.raises(ValueError,match='stale source_registry_sha256'):support.source_for_record(cfg,row)
    verify.run(cfg)
    assert not read_jsonl(cfg.run_dir/'verified.jsonl')
    assert read_jsonl(cfg.run_dir/'pending_review.jsonl')[0]['pending_reason']=='source_admission_unresolved'


def test_legacy_configuration_remains_opt_in(setup):
    cfg,r=setup;cfg.seeds.registry=None
    assert admitted_sources(cfg) is None
    train,held=load_chunks(cfg);assert len(train)==1 and 'source_registry_sha256' not in train[0]


@pytest.mark.parametrize('fault', ['old_text','missing_binding','old_prompt','old_scenario','duplicate','unplanned'])
def test_cached_question_cannot_expose_stale_source_before_teacher(setup,monkeypatch,fault):
    cfg,r=setup
    job=generate.question_jobs(cfg)[0]
    row={**job,'question':'Synthetic question?', 'scenario':generate._scenario(job),
         'prompt_version':generate.PROMPT_VERSION,'mock_version':generate.MOCK_VERSION}
    if fault=='old_text':row['text']='UNIQUE_HELD_OUT_RESUME_MARKER'
    elif fault=='missing_binding':row.pop('source_registry_sha256')
    elif fault=='old_prompt':row['prompt_version']='old'
    elif fault=='old_scenario':row['scenario']={'invented':'UNIQUE_HELD_OUT_RESUME_MARKER'}
    elif fault=='unplanned':row['id']='outside-plan'
    path=cfg.run_dir/'questions.jsonl';write_jsonl(path,[row,row] if fault=='duplicate' else [row])
    original=path.read_bytes();constructed=[]
    def forbidden(*args,**kwargs):constructed.append(True);raise AssertionError('teacher must not be constructed')
    monkeypatch.setattr(generate,'Teacher',forbidden)
    with pytest.raises(ValueError):generate.run(cfg)
    assert not constructed and path.read_bytes()==original


def test_identical_registry_resume_sends_only_missing_answer(setup,monkeypatch):
    cfg,r=setup;job=generate.question_jobs(cfg)[0]
    row={**job,'question':'Synthetic question?', 'scenario':generate._scenario(job),
         'prompt_version':generate.PROMPT_VERSION,'mock_version':generate.MOCK_VERSION}
    write_jsonl(cfg.run_dir/'questions.jsonl',[row])
    FakeTeacher.requests=[];monkeypatch.setattr(generate,'Teacher',FakeTeacher)
    generate.run(cfg)
    assert [request[0] for request in FakeTeacher.requests]==['answer']
    generate.run(cfg)
    assert len(FakeTeacher.requests)==1


def test_direct_training_rechecks_source_before_tokenizer_or_model(setup,monkeypatch):
    cfg,r=setup;FakeTeacher.requests=[];monkeypatch.setattr(generate,'Teacher',FakeTeacher)
    generate.run(cfg);rows=read_jsonl(cfg.run_dir/'generated.jsonl')
    write_jsonl(cfg.run_dir/'verified.jsonl',rows)
    assert train.admitted_training_rows(cfg)==rows
    r['documents'][0]['source_version']='a-new-version';save(cfg,r)
    # This production entry point fails before even importing its runtime stack.
    with pytest.raises(ValueError,match='stale source_registry_sha256'):train.run(cfg)
