"""Read-only exact evaluation checks; no model requests or generated-code execution."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,unicodedata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def norm(text):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))
def record_hash(row):return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate(here=HERE,root=ROOT,*,check_manifest=True):
    dev=rows(here/'development_normal.jsonl');tools=rows(here/'development_tools.jsonl');final=rows(here/'final_normal.jsonl')
    calibration=rows(root/'data/eval/judge_calibration.jsonl')
    allrows=dev+tools+final;allids={r['id'] for r in allrows}
    assert (len(dev),len(tools),len(final),len(calibration))==(106,23,22,24)
    assert len(allids)==151 and len({r['id'] for r in allrows+calibration})==175,'duplicate IDs'
    assert len({norm(r['question']) for r in allrows+calibration})==175,'normalized duplicate questions'
    for r in dev+final:
        assert set(r)=={'id','doc_id','task','question','reference'},r['id']
        assert r['task'] in {'concept','howto','debug','script'} and r['reference'].strip() and r['question'].strip()
    assert dev[:66]==rows(root/'data/eval/dev_v3/normal.jsonl'),'inherited normal rows changed'
    assert tools[:16]==rows(root/'data/eval/dev_v3/tools.jsonl'),'inherited tool rows changed'
    assert sum(r['id']=='req-016' for r in dev)==1
    for r in tools:
        assert r['expect'] in {'call','no_call'} and r['task']=='tool'
        assert r['scoring_protocol']=='tool-first-response-v2' and r['application_policy']=='hpc-tools-clarify-first-v1'
        assert 'allow_lookup' not in r
    metadata=json.loads((here/'item_metadata.json').read_text())['items']
    assert {m['id'] for m in metadata}==allids and len(metadata)==151
    records={r['id']:r for r in allrows};span_count=0
    for m in metadata:
        r=records[m['id']]
        assert m['record_sha256']==record_hash(r) and m['question_sha256']==hashlib.sha256(r['question'].encode()).hexdigest()
        assert m['training_eligible'] is False
        assert m['split']==('final' if r in final else 'development')
        assert m['evidence'] and m['scoring'] and m['review']
        for e in m['evidence']:
            base=here if e['root']=='release' else root
            path=(base/e['path']).resolve();assert path.is_relative_to(base.resolve())
            assert sha(path)==e['sha256'],e['path']
            if 'start_line' in e:
                assert '\n'.join(path.read_text().splitlines()[e['start_line']-1:e['end_line']])==e['quote'],m['id']
                span_count+=1
    source=json.loads((here/'source_manifest.json').read_text())
    for e in source['acquired']:
        assert sha(here/'sources'/e['local_path'].removeprefix('raw/'))==e['sha256']
    assert source['missing_original_provenance_fields']==18
    if check_manifest:
        manifest=json.loads((here/'manifest.json').read_text())
        for e in manifest['bindings']:
            base=here if e['root']=='release' else root
            assert sha(base/e['path'])==e['sha256'],e['path']
        assert manifest['calibration_counted_in_total'] is False and manifest['diagnostics_counted_in_total'] is False
    return {'status':'PASS','normal_development_ready':len(dev),'first_response_tool_development_ready':len(tools),
            'final_normal_ready':len(final),'total_ready':len(allrows),'draft_pending':0,'calibration_pairs_separate':len(calibration),
            'duplicate_ids':0,'normalized_duplicate_questions':0,'json_schema_errors':0,'references_nonempty':128,
            'metadata_records':len(metadata),'exact_evidence_spans':span_count,'acquired_source_hashes':len(source['acquired']),
            'inherited_82_unchanged':True,'old_26_added_again':False,'normal_tasks':dict(Counter(r['task'] for r in dev+final)),
            'new_training_records':0,'semantic_guarantee':False,'model_inference_or_compilation':'NOT RUN'}

if __name__=='__main__':print(json.dumps(validate(),indent=2))
