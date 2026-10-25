"""Recheck this release's exact schema/bindings; no model requests or tool execution."""
import hashlib
import json
from pathlib import Path
import unicodedata
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    normal, tools = rows(HERE/'normal.jsonl'), rows(HERE/'tools.jsonl')
    old_normal, old_tools = rows(ROOT/'data/eval/eval_all_v2.jsonl'), rows(ROOT/'data/eval/eval_tools.jsonl')
    calibration = rows(ROOT/'data/eval/judge_calibration.jsonl')
    changes = json.loads((HERE/'corrections.json').read_text())['changes']
    all_items = normal+tools+calibration
    norm = lambda s:' '.join(unicodedata.normalize('NFKC',s).casefold().split())
    assert len({r['id'] for r in all_items}) == len(all_items), 'duplicate IDs across slices'
    assert len({norm(r['question']) for r in all_items}) == len(all_items), 'normalized duplicates across slices'
    assert all(set(r)=={'id','doc_id','task','question','reference'} for r in normal)
    assert all(r['doc_id'] in {'slurm_job_arrays','slurm_requeue_signals'} and
               r['task'] in {'concept','howto','debug','script'} and r['reference'].strip() for r in normal)
    assert len(normal)==66 and len(tools)==16 and len(calibration)==24
    assert [{k:v for k,v in r.items() if k!='reference'} for r in normal] == [
        {k:v for k,v in r.items() if k!='reference'} for r in old_normal]
    assert [r['question'] for r in tools]==[r['question'] for r in old_tools]
    assert {a['id'] for a,b in zip(old_normal,normal) if a!=b}=={c['id'] for c in changes}
    assert sum(r['id']=='req-016' for r in normal)==1
    assert all(r['application_policy']=='hpc-tools-clarify-first-v1' and
               r['scoring_protocol']=='tool-first-response-v2' for r in tools)
    assert not any('allow_lookup' in r for r in tools)
    script=tools[8]['question'].split('\n\n',1)[1]
    assert tools[8]['expected_call']['arguments']=={'script':[script],'test_only':[True]}
    for c in changes:
        assert sha(ROOT/c['source']['path'])==c['source']['sha256']
        assert c['source']['quote'] in (ROOT/c['source']['path']).read_text()
        assert next(r['reference'] for r in normal if r['id']==c['id'])==c['reference_after']
    manifest=json.loads((HERE/'manifest.json').read_text())
    for group in ['inputs','outputs','code_bindings']:
        for binding in manifest[group]:
            assert sha(ROOT/binding['path'])==binding['sha256'], binding['path']
    return {'normal_ready':len(normal),'tool_ready_first_response_scope':len(tools),
            'draft_pending':0,'calibration_pairs_separate':len(calibration),'unique_ids_across_slices':len(all_items),
            'exact_duplicate_questions':0,'normalized_duplicate_questions':0,'json_errors':0,
            'normal_by_doc':dict(Counter(r['doc_id'] for r in normal)),
            'normal_by_task':dict(Counter(r['task'] for r in normal)),
            'tool_decisions':dict(Counter(r['expect'] for r in tools)),
            'reference_changes_only':[r['id'] for r in changes],
            'normal_cached_inputs_compatible':66,'tool_cached_inputs_incompatible':16,
            'unknown_original_provenance_fields':18,
            'semantic_limitation':'Exact checks do not prove entailment or semantic independence; inherited overlap decisions and model review remain fallible.'}


if __name__ == '__main__':
    print(json.dumps(validate(),indent=2))
