"""Synthetic certificates exercise the real admission path, never a model.

These fixtures deliberately attest invented records. They test binding mechanics,
not the truth of a semantic review or a real token/mask check.
"""
import builtins
import copy
import hashlib
import json
from pathlib import Path

import pytest

from distillkit import generate, records, train
from distillkit.config import Config
from distillkit.seeds import load_chunks


TEXT = 'Synthetic source: a reported count does not establish a failure cause.\n'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + '\n').encode()


def family(split='train', parents=()):
    return {'split': split, 'parents': list(parents), 'lineage_reviewed': True}


class Bundle:
    """Write a reviewed-input fixture without calling generation or config dotenv loading."""

    def __init__(self, root):
        self.root = root
        self.files = set()
        self.origins = []
        self.cfgs = []
        self.rows = []

    def put(self, path, value, *, raw=False):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if raw else encoded(value))
        self.files.add(path)
        return path

    def read(self, path):
        return json.loads((self.root / path).read_bytes())

    def hash(self, path):
        return sha((self.root / path).read_bytes())

    def origin(self, ident, *, legacy=False, doc='fixture', source=TEXT,
               question=None, qid=None, gold=True, family_name='fixture-family',
               extra_families=None, extra_documents=(), registry_edit=None,
               row_edit=None, tools=None, mode='prose', cpu_style='current'):
        base = ident + '/run'
        corpus = base + '/corpus'
        seed = self.put(corpus + '/seeds/' + doc + '.md', source.encode(), raw=True)
        source_hash = self.hash(seed)
        registry_path = corpus + '/registry.json'
        registry = {
            'version': 'source-family-registry-v1',
            'families': {family_name: family(), **(extra_families or {})},
            'documents': [{
                'doc_id': doc, 'path': doc + '.md', 'sha256': source_hash,
                'family': family_name, 'approved_for_training': True,
                'origin': 'synthetic unit-test source', 'source_version': 'fixture-v1',
                'permission_evidence': 'Synthetic test fixture only; not a real source permission.',
                'admission_review_sha256': 'a' * 64,
            }, *extra_documents],
        }
        if not legacy:
            self.put(registry_path, registry)
        source_manifest = self.put(corpus + '/source_manifest.json', {'documents': [{
            'doc_id': doc, 'sha256': source_hash, 'split': 'train',
            'family': family_name,
            'permission_evidence': 'Synthetic fixture notice; upstream unknowns remain unknown.',
        }]})
        self.put(corpus + '/notices/fixture.txt', b'Synthetic notice retained byte for byte.\n', raw=True)
        endpoint = {'base_url': 'http://invalid.test/v1', 'model': 'synthetic-teacher-no-inference'}
        cfg = Config.model_validate({
            'run_dir': str(self.root / base / 'outputs'),
            'seeds': {'dir': str(self.root / corpus / 'seeds'), 'eval_docs': ['held'],
                      'chunk_chars': 20000,
                      'registry': None if legacy else str(self.root / registry_path)},
            'teacher': endpoint, 'judge': endpoint,
            'generate': {'questions_per_chunk': 1, 'answers_per_question': 1,
                         'personas': ['synthetic user'], 'task_types': ['concept'],
                         'gold_dir': str(self.root / corpus / 'gold') if gold else None},
            'verify': {}, 'student': {'model': 'synthetic-student', 'system_prompt': 'Synthetic system.'},
            'train': {'max_length': 2048}, 'eval': {'file': str(self.root / 'unused-eval.jsonl')},
        })
        chunks, _ = load_chunks(cfg)
        chunk = chunks[0]
        # Mutating registry bytes after chunk creation models stale record metadata,
        # while the later manifest/certificate still bind the changed actual file.
        if registry_edit is not None:
            registry_edit(registry)
            self.put(registry_path, registry)
        gold_files = []
        if gold:
            provenance = {'schema_version': 1, 'examples': {}}
            for name in ('prose.json', 'tool_trace.json'):
                path = self.put(corpus + '/gold/' + name, {'messages': [
                    {'role': 'user', 'content': 'Synthetic example?'},
                    {'role': 'assistant', 'content': 'A count alone does not establish the cause.'},
                ]})
                provenance['examples'][name] = {
                    'sha256': self.hash(path),
                    'sources': [{'doc_id': doc, 'split': 'train', 'sha256': source_hash}],
                }
                gold_files.append({'path': path, 'sha256': self.hash(path)})
            path = self.put(corpus + '/gold/provenance.json', provenance)
            gold_files.append({'path': path, 'sha256': self.hash(path)})
        config_path = self.put(base + '/effective_config.json', cfg.model_dump(mode='json'))
        corpus_hashes = {p.removeprefix(base + '/'): self.hash(p)
                         for p in sorted(self.files) if p.startswith(corpus + '/')}
        if legacy:
            manifest = {'revision': 'synthetic-code-v1', 'effective_config': self.read(config_path),
                        'corpus_hashes': corpus_hashes}
        else:
            manifest = {'code_revision': 'synthetic-code-v1',
                        'bindings': {'effective_config.json': self.hash(config_path), **corpus_hashes}}
        manifest_path = self.put(base + '/manifest.json', manifest)
        question = question or ('What does the synthetic count establish in ' + ident + '?')
        qid = qid or ident + '/question'
        q = {**chunk, 'id': qid, 'question': question, 'persona': 'synthetic user',
             'task': 'concept', 'mode': mode}
        row = {k: v for k, v in q.items() if k != 'text'}
        row.update(id=qid + '/s0', sample=0, teacher='synthetic-teacher-no-inference', messages=[
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': 'The count alone does not establish a failure cause.'},
        ])
        if tools is not None:
            row['tools'] = tools
        generation_path = None
        if not legacy:
            raw_cfg = self.read(config_path)
            inputs = {'generate': raw_cfg['generate'], 'seeds': raw_cfg['seeds'],
                      'teacher': {k: v for k, v in raw_cfg['teacher'].items()
                                  if k not in ('base_url', 'api_key_env')},
                      'source_registry_sha256': self.hash(registry_path),
                      'gold_files': {Path(g['path']).name: g['sha256'] for g in gold_files}}
            binding = digest(inputs)
            generation_path = self.put(base + '/outputs/generation_manifest.json',
                                       {'binding_sha256': binding, 'inputs': inputs})
            q['generation_manifest_sha256'] = row['generation_manifest_sha256'] = binding
            row.update(source_kind='chunk', source_sha256=sha(chunk['text'].encode()))
        if row_edit is not None:
            row_edit(row)
        generated = self.put(base + '/outputs/generated.jsonl', encoded(row), raw=True)
        questions = self.put(base + '/outputs/questions.jsonl', encoded(q), raw=True)
        selected = self.put(ident + '/accepted.jsonl', encoded(row), raw=True)
        origin = {
            'origin_id': ident, 'admission_kind': 'legacy-question-chunk-certificate-v1' if legacy else 'original-source-registry-v1',
            'count': 1, 'record_ids': [row['id']], 'records': selected, 'generated': generated,
            'questions': questions, 'config': config_path, 'run_root': base,
            'run_manifest': manifest_path, 'source_manifest': source_manifest,
            'generation_revision': 'synthetic-code-v1', 'seed_names': [doc + '.md'],
            'gold_files': gold_files, 'registry': None if legacy else registry_path,
        }
        if legacy:
            inputs = {'original-generated': generated, 'original-questions': questions}
            review = {'input_hashes': {k: self.hash(p) for k, p in inputs.items()}, 'rows': [{
                'id': row['id'], 'training_record_eligible': True,
                'quality_accepted': True, 'pipeline_decision': 'pass',
                'source_document_sha256': source_hash,
                'source_chunk_sha256': sha(chunk['text'].encode()),
            }]}
            origin['legacy_review'] = self.put(ident + '/legacy-review.json', review)
            origin['legacy_review_bindings'] = inputs
            origin['cpu_evidence'] = self.put(ident + '/cpu.json', {
                'all_passed': True, 'eligible_sha256': self.hash(selected),
            })
        else:
            origin['generation_manifest'] = generation_path
            release = self.put(ident + '/release.json', {
                'accepted_ids': [row['id']], 'accepted_sha256': self.hash(selected),
            })
            cpu = self.put(ident + '/cpu.json', {
                'rows': 1, 'admission': 'PASS', 'sidecar_schema': 'PASS', 'roundtrip': 'PASS',
                'tokens': {'PASS': 1, 'FILTERED': 0, 'PENDING': 0},
                'raw_subset_sha256': self.hash(selected), 'original_config_sha256': self.hash(config_path),
            })
            if cpu_style == 'historical':
                cert = {'status': 'DELIVERED_UNDER_DOCUMENTED_CHECKS',
                        'raw_subset_sha256': self.hash(selected),
                        'immutable_candidate_manifest_sha256': self.hash(release),
                        'final_cpu_result_sha256': self.hash(cpu)}
            else:
                cert = {'status': 'training_eligible_under_recorded_checks',
                        'accepted_sha256': self.hash(selected), 'manifest_sha256': self.hash(release),
                        'cpu_evidence': {'sha256': self.hash(cpu)}}
            origin['release_manifest'], origin['cpu_evidence'] = release, cpu
            origin['certificate'] = self.put(ident + '/certificate.json', cert)
            sidecar = {'id': row['id'], 'run_id': ident,
                       'bindings': {'record_sha256': digest(row), 'source_sha256': row['source_sha256'],
                                    'source_registry_sha256': self.hash(registry_path),
                                    'config_sha256': self.hash(config_path)},
                       'decision': {'training': 'accept', 'quality': 'accept'},
                       'release_manifest': {'sha256': self.hash(release)}}
            origin['sidecars'] = self.put(ident + '/sidecars.jsonl', encoded(sidecar), raw=True)
        self.origins.append(origin)
        self.cfgs.append(cfg)
        self.rows.append(row)
        return origin

    def finish(self, *, tools=None):
        first = self.cfgs[0]
        self.manifest = {
            'version': 'multi-origin-training-inputs-v1', 'root': '.',
            'input_bindings': {p: self.hash(p) for p in sorted(self.files)},
            'expected_rows': len(self.rows),
            'contract': {'student_model': first.student.model,
                         'system_prompt_sha256': sha(first.student.system_prompt.encode()),
                         'max_length': first.train.max_length,
                         'records_adapter_sha256': sha(Path(records.__file__).read_bytes()),
                         'tool_schemas_sha256': digest(tools) if tools is not None else None},
            'origins': copy.deepcopy(self.origins),
        }
        self.path = self.root / 'training-inputs.json'
        self.caller = first.model_copy(deep=True)
        self.caller.run_dir = self.root / 'new-training-run'
        return self.save()

    def save(self, *, rebind=False):
        if rebind:
            self.manifest['input_bindings'] = {p: self.hash(p) for p in sorted(self.files)}
        self.path.write_bytes(encoded(self.manifest))
        # Use the actual typed Config parser, not unchecked nested model_copy updates.
        raw = self.caller.model_dump(mode='json')
        raw['train']['input_manifest'] = {'path': str(self.path), 'sha256': sha(self.path.read_bytes())}
        self.caller = Config.model_validate(raw)
        return self.caller


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    # Importing the generation module for _gold must not construct a client.
    def forbidden(*args, **kwargs):
        raise AssertionError('admission must not initialize a teacher or read dotenv')
    monkeypatch.setattr(generate, 'Teacher', forbidden)
    monkeypatch.setattr('distillkit.config.load_dotenv', forbidden)
    return Bundle(tmp_path)


def test_multiple_registered_origins_use_actual_guards_and_preserve_rows(bundle):
    bundle.origin('first', cpu_style='historical')
    bundle.origin('second', doc='other', source='A second synthetic source.\n', family_name='other-family')
    cfg = bundle.finish()
    before = {p: bundle.hash(p) for p in bundle.files}
    rows = train.admitted_training_rows(cfg, evidence_dir=cfg.run_dir)
    assert rows == bundle.rows
    assert [r['source_registry_sha256'] for r in rows] == [bundle.hash(o['registry']) for o in bundle.origins]
    assert {p: bundle.hash(p) for p in bundle.files} == before
    lineage = [json.loads(line) for line in (cfg.run_dir / 'training_inputs.lineage.jsonl').read_text().splitlines()]
    assert [r['origin_id'] for r in lineage] == ['first', 'second']
    assert [r['original_line_sha256'] for r in lineage] == [bundle.hash(o['records']) for o in bundle.origins]
    report = json.loads((cfg.run_dir / 'training_inputs.admission.json').read_text())
    assert report['rows'] == 2 and report['semantic_review_rerun'] is False
    assert report['raw_rows_modified'] is report['legacy_registry_metadata_added'] is False
    assert train.admitted_training_rows(cfg, evidence_dir=cfg.run_dir) == rows


def test_certified_legacy_is_narrow_and_not_retroactively_restamped(bundle):
    bundle.origin('registered')
    bundle.origin('historical', legacy=True)
    cfg = bundle.finish()
    rows = train.admitted_training_rows(cfg, evidence_dir=cfg.run_dir)
    assert rows == bundle.rows
    assert 'source_registry_sha256' in rows[0] and 'source_registry_sha256' not in rows[1]
    assert 'source_sha256' not in rows[1] and 'source_kind' not in rows[1]
    assert records.training_example(rows[1], cfg.student.system_prompt)['messages'][0]['role'] == 'system'


def test_unknown_legacy_has_no_registered_source_corroboration(bundle):
    bundle.origin('unregistered', legacy=True)
    with pytest.raises(ValueError, match='registered-origin corroboration'):
        train.admitted_training_rows(bundle.finish())


@pytest.mark.parametrize('defect', ['review_absent', 'wrong_selection', 'quality_reject',
                                  'missing_review_input', 'chunk_hash', 'retroactive_registry'])
def test_legacy_certificate_cannot_admit_arbitrary_rows(bundle, defect):
    bundle.origin('registered')
    origin = bundle.origin('legacy', legacy=True,
                           row_edit=(lambda r: r.update(source_registry_sha256='b' * 64))
                           if defect == 'retroactive_registry' else None)
    bundle.finish()
    if defect == 'review_absent':
        del bundle.manifest['origins'][1]['legacy_review']
    elif defect == 'missing_review_input':
        del bundle.manifest['origins'][1]['legacy_review_bindings']['original-generated']
    elif defect != 'retroactive_registry':
        review = bundle.read(origin['legacy_review'])
        row = review['rows'][0]
        if defect == 'wrong_selection':
            row['training_record_eligible'] = False
        elif defect == 'quality_reject':
            row['quality_accepted'] = False
        else:
            row['source_chunk_sha256'] = '0' * 64
        bundle.put(origin['legacy_review'], review)
    with pytest.raises(ValueError):
        train.admitted_training_rows(bundle.save(rebind=True))


@pytest.mark.parametrize('target', ['config', 'registry', 'gold', 'seed', 'notice', 'source_manifest'])
def test_changed_original_bytes_fail_before_runtime_import(bundle, monkeypatch, target):
    origin = bundle.origin('registered')
    cfg = bundle.finish()
    path = {'gold': origin['gold_files'][0]['path'],
            'seed': origin['run_root'] + '/corpus/seeds/fixture.md',
            'notice': origin['run_root'] + '/corpus/notices/fixture.txt'}.get(target, origin.get(target))
    with (bundle.root / path).open('ab') as file:
        file.write(b' ')
    imports = []
    original_import = builtins.__import__
    def monitor(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'trl', 'datasets', 'peft'}:
            imports.append(name)
            raise AssertionError('runtime imported before admission')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', monitor)
    with pytest.raises(ValueError, match='input hash mismatch'):
        train.run(cfg)
    assert imports == []
    assert not cfg.run_dir.exists()


def test_same_seed_but_changed_original_registry_does_not_restamp_row(bundle):
    bundle.origin('registered', registry_edit=lambda r: r['documents'][0].update(source_version='fixture-v2'))
    with pytest.raises(ValueError, match='stale source_registry_sha256'):
        train.admitted_training_rows(bundle.finish())


@pytest.mark.parametrize('defect', ['config', 'gold', 'source', 'generation', 'cpu', 'sidecar'])
def test_rebinding_outer_manifest_does_not_bypass_inner_original_evidence(bundle, defect):
    origin = bundle.origin('registered')
    bundle.finish()
    if defect == 'config':
        value = bundle.read(origin['config']); value['teacher']['temperature'] = 0.123
        bundle.put(origin['config'], value)
    elif defect == 'gold':
        value = bundle.read(origin['gold_files'][0]['path']); value['messages'][-1]['content'] = 'Changed.'
        bundle.put(origin['gold_files'][0]['path'], value)
    elif defect == 'source':
        bundle.put(origin['run_root'] + '/corpus/seeds/fixture.md', b'Changed source.\n', raw=True)
    elif defect == 'generation':
        value = bundle.read(origin['generation_manifest']); value['inputs']['teacher']['temperature'] = 0.123
        bundle.put(origin['generation_manifest'], value)
    elif defect == 'cpu':
        value = bundle.read(origin['cpu_evidence']); value['tokens']['PENDING'] = 1
        bundle.put(origin['cpu_evidence'], value)
    else:
        value = bundle.read(origin['sidecars']); value['decision']['quality'] = 'reject'
        bundle.put(origin['sidecars'], encoded(value), raw=True)
    with pytest.raises(ValueError):
        train.admitted_training_rows(bundle.save(rebind=True))


@pytest.mark.parametrize('kind', ['exact', 'normalized', 'id'])
def test_duplicates_are_rejected_across_original_certified_subsets(bundle, kind):
    bundle.origin('first', question='What does this synthetic count mean?', qid='shared' if kind == 'id' else None)
    bundle.origin('second', question=('  WHAT does this SYNTHETIC count mean?\n' if kind == 'normalized'
                                    else 'What does this synthetic count mean?'),
                  qid='shared' if kind == 'id' else None)
    with pytest.raises(ValueError, match='duplicate'):
        train.admitted_training_rows(bundle.finish())


def test_ancestor_only_declared_final_in_another_origin_is_forbidden(bundle):
    bundle.origin('first', family_name='shared-ancestor')
    bundle.origin('second', doc='other', source='A different source.\n', family_name='other-family',
                  extra_families={'shared-ancestor': family(), 'future-final': family('final', ['shared-ancestor'])})
    with pytest.raises(ValueError, match='combined origin source is held out'):
        train.admitted_training_rows(bundle.finish())


def test_conflicting_family_split_is_not_resolved_by_last_origin(bundle):
    bundle.origin('first', extra_families={'historical-family': family()})
    bundle.origin('second', extra_families={'historical-family': family('final')})
    with pytest.raises(ValueError, match='conflicting source-family declarations'):
        train.admitted_training_rows(bundle.finish())


def test_equal_bytes_of_protected_document_in_another_registry_are_forbidden(bundle):
    bundle.origin('first')
    protected = {'doc_id': 'held', 'path': 'held.md', 'sha256': sha(TEXT.encode()),
                 'family': 'reserved', 'approved_for_training': False,
                 'origin': 'synthetic final-only fixture', 'source_version': 'fixture-v1',
                 'permission_evidence': 'synthetic fixture only', 'admission_review_sha256': 'b' * 64}
    bundle.origin('second', doc='other', source='Distinct training source.\n', family_name='other-family',
                  extra_families={'reserved': family('final')}, extra_documents=[protected])
    with pytest.raises(ValueError, match='combined origin source is held out'):
        train.admitted_training_rows(bundle.finish())


def test_caller_eval_docs_cannot_be_removed_by_original_configs(bundle):
    bundle.origin('first')
    cfg = bundle.finish()
    cfg.seeds.eval_docs.append('fixture')
    with pytest.raises(ValueError, match='combined origin source is held out'):
        train.admitted_training_rows(cfg)


def _descendant_alias_config(bundle, parent_split):
    bundle.origin('alias-origin', doc='alias', source=TEXT, family_name='unrelated-train')
    protected = {'doc_id': 'reserved-child', 'path': 'reserved-child.md',
                 'sha256': sha(TEXT.encode()), 'family': 'protected-child',
                 'approved_for_training': False, 'origin': 'synthetic protected metadata',
                 'source_version': 'fixture-v1', 'permission_evidence': 'synthetic fixture only',
                 'admission_review_sha256': 'b' * 64}
    bundle.origin('other-origin', doc='other', source='Distinct selected source.\n',
                  family_name='other-train', extra_families={
                      'reserved-parent': family(parent_split),
                      'protected-child': family('train', ['reserved-parent'])},
                  extra_documents=[protected])
    return bundle.finish()


@pytest.mark.parametrize('held_split', ['development', 'final'])
def test_nonselected_held_descendant_hash_blocks_unrelated_alias_before_runtime(bundle, monkeypatch, held_split):
    cfg = _descendant_alias_config(bundle, held_split)
    original_import = builtins.__import__
    runtime_imports = []
    def forbid_runtime(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'trl', 'datasets', 'peft'}:
            runtime_imports.append(name)
            raise AssertionError('held-out alias reached training runtime import')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', forbid_runtime)
    with pytest.raises(ValueError, match='combined origin source is held out'):
        train.run(cfg)
    assert runtime_imports == []
    assert not cfg.run_dir.exists()


def test_nonselected_all_train_descendant_does_not_forbid_allowed_alias(bundle):
    cfg = _descendant_alias_config(bundle, 'train')
    assert train.admitted_training_rows(cfg) == bundle.rows


@pytest.mark.parametrize('defect', ['purpose', 'ineligible', 'runtime_guard', 'adapter_ready', 'unfinished', 'source_hash'])
def test_raw_record_guard_and_source_binding_use_production_helpers(bundle, defect):
    def alter(row):
        if defect == 'purpose': row['purpose'] = 'diagnostic_only'
        elif defect == 'ineligible': row['training_eligible'] = False
        elif defect == 'runtime_guard': row['messages'][-1]['origin'] = 'runtime_guard'
        elif defect == 'adapter_ready': row['messages'].insert(0, {'role': 'system', 'content': 'Already adapted.'})
        elif defect == 'unfinished': row['messages'][-1] = {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'fixture'}}]}
        else: row['source_sha256'] = '0' * 64
    bundle.origin('first', row_edit=alter)
    with pytest.raises(ValueError):
        train.admitted_training_rows(bundle.finish())


def test_raw_tool_arguments_are_preserved_while_adapter_uses_json_strings(bundle):
    schemas = [{'name': 'synthetic_read', 'parameters': {'type': 'object', 'properties': {'key': {'type': 'string'}}}}]
    def trace(row):
        row['messages'][1:1] = [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call1', 'type': 'function',
                'function': {'name': 'synthetic_read', 'arguments': {'key': 'fixture'}}}]},
            {'role': 'tool', 'tool_call_id': 'call1', 'content': '{"count": 1}'},
        ]
    bundle.origin('first', tools=schemas, mode='call', row_edit=trace)
    cfg = bundle.finish(tools=schemas)
    rows = train.admitted_training_rows(cfg)
    arguments = rows[0]['messages'][1]['tool_calls'][0]['function']['arguments']
    assert arguments == {'key': 'fixture'}
    adapted = records.training_example(rows[0], cfg.student.system_prompt)
    assert json.loads(adapted['messages'][2]['tool_calls'][0]['function']['arguments']) == arguments
    assert rows == bundle.rows


@pytest.mark.parametrize('field', ['system_prompt', 'student_model', 'max_length', 'adapter', 'tools'])
def test_changed_common_training_contract_requires_new_checks(bundle, field):
    schemas = [{'name': 'synthetic_read', 'parameters': {'type': 'object', 'properties': {}}}]
    bundle.origin('first', tools=schemas if field == 'tools' else None)
    cfg = bundle.finish(tools=schemas if field == 'tools' else None)
    if field == 'system_prompt': cfg.student.system_prompt += ' Changed.'
    elif field == 'student_model': cfg.student.model = 'different-student'
    elif field == 'max_length': cfg.train.max_length -= 1
    elif field == 'adapter':
        bundle.manifest['contract']['records_adapter_sha256'] = '0' * 64; cfg = bundle.save()
    else:
        bundle.manifest['contract']['tool_schemas_sha256'] = '0' * 64; cfg = bundle.save()
    with pytest.raises(ValueError, match='contract'):
        train.admitted_training_rows(cfg)


@pytest.mark.parametrize('defect', ['manifest_hash', 'duplicate_origin', 'extra_field', 'escape', 'secret_path', 'symlink'])
def test_manifest_is_explicit_strict_and_confined(bundle, defect, tmp_path):
    bundle.origin('first')
    cfg = bundle.finish()
    if defect == 'manifest_hash':
        bundle.path.write_bytes(bundle.path.read_bytes() + b' ')
    elif defect == 'duplicate_origin':
        bundle.manifest['origins'].append(copy.deepcopy(bundle.manifest['origins'][0])); cfg = bundle.save()
    elif defect == 'extra_field':
        bundle.manifest['skip_checks'] = True; cfg = bundle.save()
    else:
        key = '../outside-fixture.json' if defect == 'escape' else '.env-fixture' if defect == 'secret_path' else 'outside-link'
        if defect == 'symlink':
            # No target is read: resolving the path alone must refuse this escape.
            (tmp_path / key).symlink_to(tmp_path.parent / 'outside-fixture')
        bundle.manifest['input_bindings'][key] = '0' * 64; cfg = bundle.save()
    with pytest.raises(ValueError):
        train.admitted_training_rows(cfg)


def test_relocated_bundle_revalidates_original_config_and_notices_without_restamping(bundle, tmp_path):
    bundle.origin('first')
    cfg = bundle.finish()
    # Original config embeds the first location. Only declared source/gold paths
    # relocate; those immutable config bytes and all certificates stay untouched.
    import shutil
    relocated = tmp_path / 'relocated'
    relocated.mkdir()
    for path in bundle.files:
        target = relocated / path; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path / path, target)
    relocated_manifest = copy.deepcopy(bundle.manifest)
    relocated_manifest['root'] = 'relocated'
    bundle.manifest = relocated_manifest
    cfg = bundle.save()
    assert train.admitted_training_rows(cfg) == bundle.rows


def test_existing_evidence_cannot_be_overwritten_by_another_input_manifest(bundle):
    bundle.origin('first')
    cfg = bundle.finish()
    train.admitted_training_rows(cfg, evidence_dir=cfg.run_dir)
    path = cfg.run_dir / 'training_inputs.admission.json'
    before = path.read_bytes()
    # The path root is semantically equivalent but binds a different manifest.
    bundle.manifest['root'] = './'
    with pytest.raises(ValueError, match='existing training input evidence differs'):
        train.admitted_training_rows(bundle.save(), evidence_dir=cfg.run_dir)
    assert path.read_bytes() == before


def test_actual_run_passes_admission_before_first_training_runtime_import(bundle, monkeypatch):
    bundle.origin('first')
    cfg = bundle.finish()
    original_import = builtins.__import__
    reached = []
    class StopBeforeRuntime(Exception):
        pass
    def stop(name, *args, **kwargs):
        if name == 'torch':
            assert (cfg.run_dir / 'training_inputs.admission.json').is_file()
            reached.append(name)
            raise StopBeforeRuntime
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', stop)
    with pytest.raises(StopBeforeRuntime):
        train.run(cfg)
    assert reached == ['torch']


@pytest.mark.parametrize('registered', [False, True])
def test_unset_manifest_preserves_single_origin_entry_point(bundle, registered):
    origin = bundle.origin('first', legacy=not registered, gold=False)
    cfg = bundle.cfgs[0]
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / 'verified.jsonl').write_bytes((bundle.root / origin['records']).read_bytes())
    assert cfg.train.input_manifest is None
    assert train.admitted_training_rows(cfg) == bundle.rows
    assert not (cfg.run_dir / 'training_inputs.admission.json').exists()


def test_json_duplicate_keys_and_missing_original_records_are_not_silently_accepted(bundle):
    origin = bundle.origin('first')
    bundle.finish()
    # Even a manifest with the updated byte hash cannot hide duplicate JSON keys.
    raw = (bundle.root / origin['records']).read_bytes()
    bundle.put(origin['records'], raw.replace(b'{', b'{"id":"shadow",', 1), raw=True)
    with pytest.raises(ValueError, match='duplicate JSON key'):
        train.admitted_training_rows(bundle.save(rebind=True))
