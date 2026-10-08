"""Batch-source reuse through production support, verification and training paths."""
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from distillkit import support, train, verify
from distillkit.io import read_jsonl, write_jsonl
from distillkit.seeds import load_chunks
from distillkit.source_registry import admitted_sources
from distillkit.source_resolver import SourceResolver
from test_source_registry import setup, save, TEXT
from test_training_inputs import bundle


def record(cfg, *, kind='chunk', index=0):
    chunk = load_chunks(cfg)[0][0]
    source = chunk['text'] if kind == 'chunk' else (cfg.seeds.dir/'fixture.md').read_text()
    q = f'Explain synthetic observation {index}.'
    return {**{k:v for k,v in chunk.items() if k != 'text'}, 'id':f'row-{index}', 'question':q,
            'mode':'prose', 'source_kind':kind, 'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'messages':[{'role':'user','content':q},{'role':'assistant','content':TEXT}]}


@pytest.mark.parametrize('registry',[True,False])
@pytest.mark.parametrize('kind',['full','chunk'])
def test_single_and_batch_sources_chunks_and_bindings_match(setup,registry,kind):
    cfg,_ = setup
    if not registry: cfg.seeds.registry=None
    rows = [record(cfg,kind=kind,index=i) for i in range(4)]
    expected = [support.source_for_record(cfg,r) for r in rows]
    resolver = SourceResolver(cfg)
    assert resolver.admitted == admitted_sources(cfg)
    assert resolver.load_chunks() == load_chunks(cfg)
    assert [support.source_for_record(cfg,r,resolver=resolver) for r in rows] == expected
    resolver.assert_unchanged()
    assert support.source_for_records(cfg,rows) == expected


@pytest.mark.parametrize('failure',['doc','chunk','kind','text_hash','registry_hash','missing_family'])
def test_invalid_records_fail_identically_without_disk_rereads(setup,failure):
    cfg,_=setup; r=record(cfg); resolver=SourceResolver(cfg)
    if failure=='doc':r['doc_id']='unknown'
    elif failure=='chunk':r['chunk_id']='fixture#999'
    elif failure=='kind':r['source_kind']='invented'
    elif failure=='text_hash':r['source_sha256']='changed'
    elif failure=='registry_hash':r['source_registry_sha256']='changed'
    else:r.pop('source_family')
    errors=[]
    for fn in (lambda:support.source_for_record(cfg,r),lambda:support.source_for_record(cfg,r,resolver=resolver)):
        with pytest.raises(ValueError) as caught:fn()
        errors.append(str(caught.value))
    assert errors[0]==errors[1]


def test_crlf_raw_registry_hash_and_universal_newline_text_match(setup):
    cfg,registry=setup
    raw=b'A source with CRLF.\r\n\r\nA second paragraph.\r\n'
    (cfg.seeds.dir/'fixture.md').write_bytes(raw)
    registry['documents'][0]['sha256']=hashlib.sha256(raw).hexdigest();save(cfg,registry)
    for kind in ('full','chunk'):
        r=record(cfg,kind=kind)
        assert support.source_for_records(cfg,[r]) == [support.source_for_record(cfg,r)]


def test_explicit_holdout_cannot_be_read_from_snapshot(setup):
    cfg,registry=setup
    cfg.seeds.eval_docs.append('fixture')
    resolver=SourceResolver(cfg)
    assert not resolver.load_chunks()[0]
    r={'doc_id':'fixture','source_kind':'full','source_sha256':hashlib.sha256(TEXT.encode()).hexdigest()}
    with pytest.raises(ValueError,match='not an admitted'):resolver.source_for_record(r)
    cfg.seeds.registry=None
    with pytest.raises(ValueError,match='evaluation-only'):SourceResolver(cfg).source_for_record(r)


@pytest.mark.parametrize('failure',['source_same_size_and_mtime','registry','added','removed','config','config_file'])
def test_changed_snapshot_detected_before_batch_acceptance(setup,failure):
    cfg,_=setup;path=cfg.seeds.dir/'fixture.md';config_path=cfg.run_dir/'config.json'
    config_path.write_text(cfg.model_dump_json());resolver=SourceResolver(cfg,config_path=config_path)
    before=path.stat()
    if failure=='source_same_size_and_mtime':
        path.write_bytes(path.read_bytes().replace(b'count',b'value'))
        os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    elif failure=='registry':cfg.seeds.registry.write_bytes(cfg.seeds.registry.read_bytes()+b' ')
    elif failure=='added':(cfg.seeds.dir/'new.md').write_text('New source.')
    elif failure=='removed':path.unlink()
    elif failure=='config':cfg.seeds.chunk_chars+=1
    else:config_path.write_bytes(config_path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='snapshot'):resolver.assert_unchanged()


def test_resolver_cannot_cross_configuration_or_mutate_its_returned_bindings(setup):
    cfg,_=setup;r=record(cfg);resolver=SourceResolver(cfg)
    selected=resolver.admitted;selected['fixture']['source_families'].append('invented')
    chunks,_=resolver.load_chunks();chunks[0]['text']='invented'
    assert resolver.source_for_record(r)==TEXT
    other=cfg.model_copy(deep=True);other.seeds.eval_docs.append('fixture')
    with pytest.raises(ValueError,match='configuration'):
        support.source_for_record(other,r,resolver=resolver)


def test_source_file_read_count_is_constant_in_batch_size(setup,monkeypatch):
    cfg,_=setup;rows=[record(cfg,index=i) for i in range(25)]
    counts=Counter();original=Path.open;paths={cfg.seeds.registry,cfg.seeds.dir/'fixture.md'}
    def opened(path,*args,**kwargs):
        if path in paths:counts[path.name]+=1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',opened)
    expected=[support.source_for_record(cfg,r) for r in rows];legacy=dict(counts);counts.clear()
    assert support.source_for_records(cfg,rows)==expected
    assert legacy=={'registry.json':50,'fixture.md':75}
    assert counts=={'registry.json':2,'fixture.md':2}


@pytest.mark.parametrize('entry',['verify','train'])
@pytest.mark.parametrize('failure',['source','registry','config'])
def test_production_paths_refuse_mid_batch_changes_before_outputs_or_return(setup,monkeypatch,entry,failure):
    cfg,_=setup;r=record(cfg);cfg.verify.require_grounding_review=False
    write_jsonl(cfg.run_dir/'generated.jsonl',[r]);write_jsonl(cfg.run_dir/'verified.jsonl',[r])
    before=(cfg.run_dir/'verified.jsonl').read_bytes()
    def mutate():
        if failure=='source':(cfg.seeds.dir/'fixture.md').write_text('Changed during this batch.')
        elif failure=='registry':cfg.seeds.registry.write_bytes(cfg.seeds.registry.read_bytes()+b' ')
        else:cfg.seeds.chunk_chars+=1
    if entry=='verify':
        original=verify.check_answer
        def checked(*args,**kwargs):
            result=original(*args,**kwargs);mutate();return result
        monkeypatch.setattr(verify,'check_answer',checked)
        action=lambda:verify.run(cfg)
    else:
        original=SourceResolver.source_for_record
        def resolved(self,row):
            result=original(self,row);mutate();return result
        monkeypatch.setattr(SourceResolver,'source_for_record',resolved)
        action=lambda:train.admitted_training_rows(cfg)
    with pytest.raises(ValueError,match='snapshot'):action()
    assert (cfg.run_dir/'verified.jsonl').read_bytes()==before
    assert not (cfg.run_dir/'rejected.jsonl').exists()
    assert not (cfg.run_dir/'pending_review.jsonl').exists()


def test_source_for_records_detects_mutation_in_row_iterator(setup):
    cfg,_=setup;r=record(cfg)
    def rows():
        yield r
        cfg.seeds.registry.write_bytes(cfg.seeds.registry.read_bytes()+b' ')
    with pytest.raises(ValueError,match='registry changed'):support.source_for_records(cfg,rows())


def test_verify_outputs_and_direct_train_keep_original_records(setup):
    cfg,_=setup;r=record(cfg);cfg.verify.require_grounding_review=False
    write_jsonl(cfg.run_dir/'generated.jsonl',[r])
    verify.run(cfg)
    expected=read_jsonl(cfg.run_dir/'verified.jsonl')
    assert len(expected)==1 and expected[0]['messages']==r['messages']
    assert train.admitted_training_rows(cfg)==expected


@pytest.mark.parametrize('failure',['added_source','effective_config'])
def test_multi_origin_rechecks_every_origin_before_admission_evidence(bundle,monkeypatch,failure):
    from distillkit import training_inputs
    bundle.origin('first',gold=False)
    bundle.origin('second',doc='other',source='A different synthetic source.',family_name='other-family',gold=False)
    cfg=bundle.finish();original=training_inputs._cross_origin
    def changed(origins,checked,inputs,caller_holdouts):
        original(origins,checked,inputs,caller_holdouts)
        first=checked[0][0]
        if failure=='added_source':(first.seeds.dir/'unlisted.md').write_text('Unexpected later addition.')
        else:first.seeds.chunk_chars+=1
    monkeypatch.setattr(training_inputs,'_cross_origin',changed)
    with pytest.raises(ValueError,match='source snapshot'):
        train.admitted_training_rows(cfg,evidence_dir=cfg.run_dir)
    assert not (cfg.run_dir/'training_inputs.admission.json').exists()
    assert not (cfg.run_dir/'training_inputs.lineage.jsonl').exists()
