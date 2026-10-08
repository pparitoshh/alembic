"""Reviewed, opt-in multi-origin raw training inputs, before model construction.

The manifest is a trust input like a source registry, not a proof of authorship,
licensing or semantic correctness. Historical records and registries are never
rewritten. No tokenized/adapter-ready alternate input is accepted here.
"""
import copy
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import Config
from . import records
from .source_registry import Registry
from .source_resolver import SourceResolver

SHA = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class _Typed(BaseModel):
    model_config = ConfigDict(extra='forbid')


class BoundFile(_Typed):
    path: str
    sha256: SHA


class Contract(_Typed):
    student_model: str
    system_prompt_sha256: SHA
    max_length: int = Field(gt=0, strict=True)
    records_adapter_sha256: SHA
    tool_schemas_sha256: SHA | None


class _Origin(_Typed):
    origin_id: str = Field(min_length=1)
    count: int = Field(gt=0, strict=True)
    record_ids: list[str] = Field(min_length=1)
    records: str
    generated: str
    questions: str
    config: str
    run_root: str
    run_manifest: str
    source_manifest: str
    generation_revision: str = Field(min_length=1)
    seed_names: list[str] = Field(min_length=1)
    gold_files: list[BoundFile]
    cpu_evidence: str


class RegisteredOrigin(_Origin):
    admission_kind: Literal['original-source-registry-v1']
    registry: str
    release_manifest: str
    certificate: str
    sidecars: str
    generation_manifest: str


class LegacyOrigin(_Origin):
    admission_kind: Literal['legacy-question-chunk-certificate-v1']
    registry: None
    legacy_review: str
    # Every original review input key maps explicitly to a bound local file.
    legacy_review_bindings: dict[str, str] = Field(min_length=1)


class InputManifest(_Typed):
    version: Literal['multi-origin-training-inputs-v1']
    root: Path  # relative to this manifest, or an explicit relocated input root
    input_bindings: dict[str, SHA] = Field(min_length=1)
    expected_rows: int = Field(gt=0, strict=True)
    contract: Contract
    origins: list[Annotated[RegisteredOrigin | LegacyOrigin,
                            Field(discriminator='admission_kind')]] = Field(min_length=1)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode())


def _require(condition, message):
    if not condition:
        raise ValueError('training input admission: ' + message)


def _parse(raw):
    def keys(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('training input admission: nonfinite JSON ' + value)
    return json.loads(raw, object_pairs_hook=keys, parse_constant=nonfinite)


def _index(rows):
    result = {}
    for row in rows:
        _require(isinstance(row, dict), 'expected record object')
        ident = row.get('id')
        _require(isinstance(ident, str) and bool(ident) and ident not in result,
                 'missing or duplicate record ID')
        result[ident] = row
    return result


class _Inputs:
    def __init__(self, root, bindings):
        self.root, self.bindings, self.cache = root.resolve(), bindings, {}
        self.source_resolvers = []

    def path(self, rel):
        path = Path(rel)
        _require(bool(rel) and not path.is_absolute() and '..' not in path.parts
                 and '\\' not in rel and not any(p.startswith('.env') for p in path.parts),
                 'unsafe relative input path ' + str(rel))
        value = (self.root / path).resolve()
        _require(value.is_relative_to(self.root), 'input path escapes declared root')
        return value

    def raw(self, rel):
        _require(rel in self.bindings, 'unbound input ' + rel)
        if rel not in self.cache:
            raw = self.path(rel).read_bytes()
            _require(_sha(raw) == self.bindings[rel], 'input hash mismatch: ' + rel)
            self.cache[rel] = raw
        return self.cache[rel]

    def json(self, rel):
        return _parse(self.raw(rel))

    def rows(self, rel):
        raw = self.raw(rel)
        _require(bool(raw) and raw.endswith(b'\n'), 'JSONL must be nonempty and newline-terminated: ' + rel)
        lines = raw.splitlines(keepends=True)
        _require(all(line.strip() for line in lines), 'empty JSONL line: ' + rel)
        rows = [_parse(line) for line in lines]
        _require(all(isinstance(r, dict) for r in rows), 'JSONL rows must be objects')
        _index(rows)
        return lines, rows


def _registered_certificate(origin, inputs, rows):
    release, cert, cpu = (inputs.json(getattr(origin, key))
                          for key in ('release_manifest', 'certificate', 'cpu_evidence'))
    bindings = inputs.bindings
    _require(release['accepted_ids'] == origin.record_ids and
             release['accepted_sha256'] == bindings[origin.records], 'release selection differs')
    _require(cert.get('status') in ('DELIVERED_UNDER_DOCUMENTED_CHECKS', 'training_eligible_under_recorded_checks'),
             'uncertified registered subset')
    _require(cert.get('raw_subset_sha256', cert.get('accepted_sha256')) == bindings[origin.records] and
             cert.get('immutable_candidate_manifest_sha256', cert.get('manifest_sha256')) == bindings[origin.release_manifest] and
             cert.get('final_cpu_result_sha256', cert.get('cpu_evidence', {}).get('sha256')) == bindings[origin.cpu_evidence],
             'release certificate binding differs')
    _require(cpu.get('rows') == len(rows) and
             all(cpu.get(k) == 'PASS' for k in ('admission', 'sidecar_schema', 'roundtrip')) and
             cpu.get('tokens') == {'PASS':len(rows), 'FILTERED':0, 'PENDING':0} and
             cpu['raw_subset_sha256'] == bindings[origin.records] and
             cpu['original_config_sha256'] == bindings[origin.config], 'CPU subset certificate differs or is not passing')
    _, sidecars = inputs.rows(origin.sidecars)
    _require([s['id'] for s in sidecars] == origin.record_ids, 'sidecar IDs/order differ')
    for row, side in zip(rows, sidecars, strict=True):
        b = side['bindings']
        _require(b['record_sha256'] == _digest(row) and
                 b['source_registry_sha256'] == bindings[origin.registry] and
                 b['source_sha256'] == row['source_sha256'] and
                 b['config_sha256'] == bindings[origin.config] and
                 side['run_id'] == origin.origin_id and
                 side['decision'].get('training') == side['decision'].get('quality') == 'accept' and
                 side['release_manifest']['sha256'] == bindings[origin.release_manifest],
                 'acceptance sidecar differs for ' + row['id'])
    return {r['id']: {'certificate_sha256':bindings[origin.certificate],
                     'acceptance_sidecar_sha256':_digest(s)} for r,s in zip(rows,sidecars,strict=True)}


def _legacy_certificate(origin, inputs, rows):
    review, cpu = inputs.json(origin.legacy_review), inputs.json(origin.cpu_evidence)
    _require(set(review['input_hashes']) == set(origin.legacy_review_bindings), 'legacy review inputs unresolved')
    for key, path in origin.legacy_review_bindings.items():
        _require(review['input_hashes'][key] == _sha(inputs.raw(path)), 'legacy review input binding differs')
    bound_review_paths = set(origin.legacy_review_bindings.values())
    _require({origin.generated, origin.questions} <= bound_review_paths, 'legacy review lacks original questions/generated binding')
    reviewed = _index(review['rows'])
    retained = {r['id'] for r in review['rows'] if r.get('training_record_eligible') is True}
    _require(retained == set(origin.record_ids) and cpu.get('all_passed') is True and
             cpu.get('eligible_sha256') == inputs.bindings[origin.records], 'legacy selection is not explicitly certified')
    for row in rows:
        r = reviewed[row['id']]
        _require(r.get('quality_accepted') is True and r.get('pipeline_decision') == 'pass',
                 'legacy row lacks recorded quality/pipeline acceptance')
        _require(not any(k in row for k in ('source_registry_sha256', 'document_sha256',
                                          'source_kind', 'source_sha256', 'source_family', 'source_families')),
                 'legacy row has retroactively added registry metadata')
    return reviewed


def _origin(origin, inputs, contract):
    """Only declared corpus path relocation; do not call load_config (.env)."""
    raw_cfg = inputs.json(origin.config)
    cfg = Config.model_validate(raw_cfg)
    _require(cfg.train.input_manifest is None, 'origin configuration cannot recurse into another manifest')
    _require(cfg.student.model == contract.student_model and
             _sha(cfg.student.system_prompt.encode()) == contract.system_prompt_sha256 and
             cfg.train.max_length == contract.max_length, 'original training contract differs')
    manifest = inputs.json(origin.run_manifest)
    _require(manifest.get('code_revision', manifest.get('revision')) == origin.generation_revision,
             'generation revision differs from original run')
    registered = isinstance(origin, RegisteredOrigin)
    _require((cfg.seeds.registry is not None) == registered, 'origin admission kind/config mismatch')
    _require(origin.source_manifest == origin.run_root + '/corpus/source_manifest.json' and
             (not registered or origin.registry == origin.run_root + '/corpus/registry.json'),
             'declared corpus relocation does not match origin paths')
    if registered:
        _require(manifest['bindings']['effective_config.json'] == inputs.bindings[origin.config], 'original config binding differs')
        generation = inputs.json(origin.generation_manifest)
        g = generation['inputs']
        _require(generation['binding_sha256'] == _digest(g) and g['generate'] == raw_cfg['generate'] and
                 g['seeds'] == raw_cfg['seeds'] and g['source_registry_sha256'] == inputs.bindings[origin.registry],
                 'original generation/config/registry binding differs')
        teacher = {k:v for k,v in raw_cfg['teacher'].items() if k not in ('base_url','api_key_env')}
        _require(g['teacher'] == teacher, 'original teacher settings differ')
    else:
        original, current = copy.deepcopy(manifest['effective_config']), copy.deepcopy(raw_cfg)
        # Historical startup chose an ephemeral serving address; no other
        # semantic/config byte difference is silently permitted.
        original['teacher'].pop('base_url', None); current['teacher'].pop('base_url', None)
        _require(original == current, 'legacy original semantic config differs')
    corpus = manifest.get('bindings', manifest.get('corpus_hashes'))
    _require(isinstance(corpus, dict), 'original corpus bindings missing')
    # Keep notices, attribution and review evidence bound, not only seed bytes.
    for rel, expected in corpus.items():
        if rel.startswith('corpus/'):
            _require(_sha(inputs.raw(origin.run_root + '/' + rel)) == expected,
                     'original corpus/gold/permission binding differs: ' + rel)
    seed_dir = inputs.path(origin.run_root + '/corpus/seeds')
    _require(len(origin.seed_names) == len(set(origin.seed_names)) and
             sorted(p.name for p in seed_dir.glob('*.md')) == sorted(origin.seed_names), 'original selected source set changed')
    for name in origin.seed_names:
        _require(Path(name).name == name and name.endswith('.md'), 'invalid seed filename')
        inputs.raw(origin.run_root + '/corpus/seeds/' + name)
    _require(bool(cfg.generate.gold_dir) == bool(origin.gold_files), 'gold configuration/declaration differs')
    expected_gold = {origin.run_root + '/corpus/gold/' + n for n in ('prose.json','tool_trace.json','provenance.json')} if cfg.generate.gold_dir else set()
    _require({g.path for g in origin.gold_files} == expected_gold and len(origin.gold_files) == len(expected_gold), 'gold files missing/duplicated')
    for g in origin.gold_files:
        _require(_sha(inputs.raw(g.path)) == g.sha256, 'gold example binding differs')
        if registered:
            _require(generation['inputs']['gold_files'][Path(g.path).name] == g.sha256,
                     'generation gold binding differs')
    relocated = cfg.model_copy(update={
        'seeds':cfg.seeds.model_copy(update={'dir':seed_dir, 'registry':inputs.path(origin.registry) if registered else None}),
        'generate':cfg.generate.model_copy(update={'gold_dir':inputs.path(origin.run_root+'/corpus/gold') if cfg.generate.gold_dir else None})})
    resolver = SourceResolver(relocated, config_path=inputs.path(origin.config))
    inputs.source_resolvers.append(resolver)
    selected = resolver.admitted
    from .generate import _gold
    for name in ('prose', 'tool_trace'):
        _gold(relocated, name)
    raw_lines, rows = inputs.rows(origin.records)
    generated_lines, generated = inputs.rows(origin.generated)
    original_lines = {r['id']:line for r,line in zip(generated,generated_lines,strict=True)}
    _require(len(rows) == origin.count == len(origin.record_ids) and
             [r['id'] for r in rows] == origin.record_ids, 'origin IDs/order/count differ')
    _require(all(original_lines.get(r['id']) == line for r,line in zip(rows,raw_lines,strict=True)),
             'subset is not byte-identical original generated lines')
    _, question_rows = inputs.rows(origin.questions)
    questions = _index(question_rows)
    documents = inputs.json(origin.source_manifest)['documents']
    docs = {d['doc_id']:d for d in documents}
    _require(len(docs) == len(documents), 'duplicate source-manifest IDs')
    certificate = (_registered_certificate if registered else _legacy_certificate)(origin,inputs,rows)
    chunks = {c['chunk_id']:c for c in resolver.load_chunks()[0]}
    from .support import source_for_record
    for row in rows:
        qid = row['id'].rsplit('/s',1)[0]
        _require(qid in questions, 'original question missing')
        q = questions[qid]
        _require(all(row[k] == q[k] for k in ('doc_id','chunk_id','persona','task','mode','question')),
                 'original question metadata differs')
        _require(row['id'] == qid + '/s' + str(row['sample']), 'answer sample identity differs')
        chunk = chunks.get(row['chunk_id'])
        _require(chunk is not None and chunk['doc_id'] == row['doc_id'] and chunk['text'] == q['text'],
                 'original question chunk does not reconstruct')
        source_hash = _sha(inputs.raw(origin.run_root+'/corpus/seeds/'+row['doc_id']+'.md'))
        doc = docs[row['doc_id']]
        doc_hashes = {doc[k] for k in ('sha256','document_sha256') if doc.get(k)}
        _require(doc_hashes == {source_hash}, 'source/permission manifest hash differs')
        if registered:
            _require(row.get('generation_manifest_sha256') == q.get('generation_manifest_sha256') == generation['binding_sha256'],
                     'record/question generation binding differs')
            source_for_record(relocated,row,resolver=resolver)
            _require(_sha(q['text'].encode()) == row['source_sha256'], 'original question source hash differs')
        else:
            r = certificate[row['id']]
            _require(doc.get('split') == 'train' and bool(doc.get('permission_evidence')) and
                     r['source_document_sha256'] == source_hash and r['source_chunk_sha256'] == _sha(q['text'].encode()),
                     'legacy source/split/permission/chunk evidence differs')
        transcript = records.messages(row)
        _require(isinstance(transcript, list) and transcript and all(isinstance(m, dict) for m in transcript) and
                 transcript[0].get('role') == 'user' and
                 transcript[0].get('content') == row['question'] and records.final_answer(row).strip(),
                 'expected complete raw teacher transcript, not adapter-ready data')
        _require(row.get('tools') is None or _digest(row['tools']) == contract.tool_schemas_sha256,
                 'tool schema differs from common training contract')
        records.training_example(row, cfg.student.system_prompt)  # actual guard/adapter, before tokenizer
    return relocated, selected, rows, raw_lines, docs, certificate


def _cross_origin(origins, checked, inputs, caller_holdouts):
    families, held_hashes, held_ids = {}, set(), set(caller_holdouts)
    registries = []
    for o,(cfg,selected,*_) in zip(origins,checked,strict=True):
        held_ids.update(cfg.seeds.eval_docs)
        if selected is not None:
            registry = Registry.model_validate(inputs.json(o.registry)); registries.append(registry)
            for name, family in registry.families.items():
                _require(name not in families or families[name] == family, 'conflicting source-family declarations: ' + name)
                families[name] = family
    def ancestors(name, stack=()):
        _require(name in families and name not in stack, 'unknown/cyclic combined source family: ' + name)
        f = families[name]
        _require(f.lineage_reviewed, 'unresolved combined family lineage: ' + name)
        return {name}.union(*(ancestors(p,(*stack,name)) for p in f.parents))
    held_families = set()
    for name,f in families.items():
        if f.split in ('development','final'):
            held_families.update(ancestors(name))
    for reg in registries:
        for d in reg.documents:
            if d.doc_id in held_ids:
                held_families.update(ancestors(d.family))
    for reg in registries:
        held_hashes.update(d.sha256 for d in reg.documents
                           if d.doc_id in held_ids or ancestors(d.family) & held_families)
    corroborated = {}
    for o,(cfg,selected,*_) in zip(origins,checked,strict=True):
        for doc_id,b in (selected or {}).items():
            lineage = ancestors(b['source_family'])
            _require(doc_id not in held_ids and not lineage & held_families and
                     all(families[f].split == 'train' for f in lineage) and b['document_sha256'] not in held_hashes,
                     'combined origin source is held out or not train: ' + doc_id)
            corroborated[(doc_id,b['document_sha256'])] = o.origin_id
    for o,(cfg,selected,rows,_,docs,certificate) in zip(origins,checked,strict=True):
        if selected is None:
            # Cover every legacy training source, including sources used only by
            # gold. Do not manufacture registry fields on historical records.
            for name in o.seed_names:
                doc_id = Path(name).stem
                if doc_id not in cfg.seeds.eval_docs:
                    h = _sha(inputs.raw(o.run_root+'/corpus/seeds/'+name))
                    _require((doc_id,h) in corroborated,
                             'legacy source lacks exact registered-origin corroboration: ' + doc_id)


def admitted_rows(cfg, *, evidence_dir=None):
    reference = cfg.train.input_manifest
    path = reference.path
    raw = path.read_bytes()
    _require(_sha(raw) == reference.sha256, 'manifest hash mismatch')
    manifest = InputManifest.model_validate(_parse(raw))
    inputs = _Inputs(path.parent / manifest.root, manifest.input_bindings)
    for rel in inputs.bindings:
        inputs.raw(rel)
    c = manifest.contract
    _require(cfg.student.model == c.student_model and _sha(cfg.student.system_prompt.encode()) == c.system_prompt_sha256 and
             cfg.train.max_length == c.max_length and _sha(Path(records.__file__).read_bytes()) == c.records_adapter_sha256,
             'caller student/template/length or record-adapter contract differs; rerun combined checks')
    origins = manifest.origins
    _require(len({o.origin_id for o in origins}) == len(origins) and len({o.records for o in origins}) == len(origins), 'duplicate origin/subset')
    checked = []
    for origin in origins:
        try:
            checked.append(_origin(origin,inputs,c))
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f'training input admission: malformed evidence for {origin.origin_id}: {exc}') from exc
    _cross_origin(origins,checked,inputs,cfg.seeds.eval_docs)
    rows = [r for _,_,batch,*_ in checked for r in batch]
    _index(rows)
    normalized = set()
    for row in rows:
        q = row.get('question')
        _require(isinstance(q,str) and bool(q.strip()), 'empty question')
        key = ' '.join(unicodedata.normalize('NFKC',q).casefold().split())
        _require(key not in normalized, 'duplicate exact/normalized question across origins')
        normalized.add(key)
    _require(len(rows) == manifest.expected_rows, 'combined row count differs')
    lineage = []
    for o,(_,_,batch,lines,_,certificate) in zip(origins,checked,strict=True):
        for n,(row,line) in enumerate(zip(batch,lines,strict=True),1):
            lineage.append({'combined_line':len(lineage)+1,'id':row['id'],'origin_id':o.origin_id,
                'original_file':o.records,'original_file_sha256':inputs.bindings[o.records],
                'original_line':n,'original_line_sha256':_sha(line),'record_sha256':_digest(row),
                'original_config_sha256':inputs.bindings[o.config],'source_registry_sha256':row.get('source_registry_sha256'),
                'generation_revision':o.generation_revision,'gold_files':[g.model_dump()for g in o.gold_files],
                'admission_kind':o.admission_kind,'preserved_evidence_sha256':_digest(certificate[row['id']]),
                'loader_example_sha256':_digest(records.training_example(row,cfg.student.system_prompt))})
    # Current helpers read files directly. Detect any mid-check replacement too.
    _require(path.read_bytes() == raw, 'manifest changed during checks')
    for rel,h in inputs.bindings.items():
        _require(_sha(inputs.path(rel).read_bytes()) == h, 'input changed during checks: ' + rel)
    for resolver in inputs.source_resolvers:
        resolver.assert_unchanged()
    if evidence_dir is not None:
        output = Path(evidence_dir); output.mkdir(parents=True,exist_ok=True)
        report = {'version':'multi-origin-training-admission-v1','manifest_sha256':reference.sha256,
                  'rows':len(rows),'contract':c.model_dump(),'semantic_review_rerun':False,
                  'raw_rows_modified':False,'legacy_registry_metadata_added':False}
        payloads = {'training_inputs.lineage.jsonl':''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in lineage),
                    'training_inputs.admission.json':json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+'\n'}
        for name,text in payloads.items():
            target = output/name
            if target.exists():
                _require(target.read_bytes() == text.encode(), 'existing training input evidence differs: ' + name)
            else:
                with target.open('xb') as f:
                    f.write(text.encode())
    return rows
