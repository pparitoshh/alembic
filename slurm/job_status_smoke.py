"""Fixed-case teacher diagnostic using production tool_trace and isolated mocks.

No generated conversations are automatically eligible for training. No .env lookup.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from distillkit.config import Config
from distillkit.generate import PROMPT_VERSION, _gold, tool_trace
from distillkit.io import JsonlAppender, read_jsonl
from distillkit.job_status_guard import WORKFLOW_VERSION
from distillkit.records import final_answer
from distillkit.teacher import Teacher
from distillkit.toolcheck import TRAINING_TRACE_POLICY_VERSION, check_trace, message_calls
from distillkit.tools import CATALOG_VERSION, MOCK_VERSION, SCHEMAS, execute, mock_session
import distillkit.generate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(root):
    manifest = json.loads((root/'manifest.json').read_text())
    for name, expected in manifest['input_hashes'].items():
        assert sha(root/name) == expected, f'Changed input: {name}'
    for name, expected in manifest['code_hashes'].items():
        assert sha(root/'code'/name) == expected, f'Changed code: {name}'
    cfg = Config.model_validate(json.loads((root/'effective_config.json').read_text()))
    assert Path(distillkit.generate.__file__).resolve().is_relative_to(root/'code')
    assert cfg.teacher.model == 'Qwen/Qwen3-32B-AWQ' and cfg.teacher.api_key_env is None
    _gold(cfg,'prose');_gold(cfg,'tool_trace')
    assert check_trace({'mode':'call',**json.loads((cfg.generate.gold_dir/'tool_trace.json').read_text())})['passed']
    plan = json.loads((root/'case_plan.json').read_text())
    assert Counter(c['mode'] for c in plan['cases']) == {'call':8,'ask':2,'none':2}
    assert len({c['id'] for c in plan['cases']}) == 12
    assert plan['mock_catalog_version'] == CATALOG_VERSION
    assert plan['source_doc_id'] not in cfg.seeds.eval_docs
    source_path=cfg.seeds.dir/(plan['source_doc_id']+'.md')
    source=source_path.read_text()
    assert all(c['doc_id']==plan['source_doc_id'] for c in plan['cases'])
    normalized=lambda q:' '.join(q.lower().split())
    protected=read_jsonl(cfg.eval.file)+read_jsonl(cfg.eval.tool_file)
    assert not {normalized(c['question']) for c in plan['cases']} & {normalized(c['question']) for c in protected}
    assert not (root/'outputs').exists(), 'fresh unused outputs required'
    with mock_session(job_ids=plan['mock_job_ids']):
        expected={c['id']:execute('job_status',{'job_id':c['expected_job_id']}) for c in plan['cases'] if c['mode']=='call'}
    return cfg,plan,source,expected


def run(root,port):
    cfg,plan,source,expected = preflight(root)
    assert os.environ.get('SLURM_JOB_ID') and port
    cfg.teacher.base_url=f'http://127.0.0.1:{port}/v1'
    experiment_attempt=json.loads((root/'manifest.json').read_text()).get('experiment_attempt',1) if (root/'manifest.json').exists() else 1
    assert experiment_attempt in (1,2)
    out=root/'outputs';out.mkdir()
    (root/'effective_runtime_config.json').write_text(cfg.model_dump_json(indent=2)+'\n')
    (root/'expected_tool_results.json').write_text(json.dumps(expected,indent=2)+'\n')
    requests=JsonlAppender(out/'request_attempts.jsonl')
    records=JsonlAppender(out/'conversations.jsonl')
    started=time.monotonic()
    def one(case):
        teacher=Teacher(cfg.teacher)
        original=teacher.client.chat.completions.create
        attempt=0
        def capture(**kwargs):
            nonlocal attempt
            attempt+=1
            e={'case_id':case['id'],'request_attempt':attempt,'started_utc':datetime.now(timezone.utc).isoformat(),
               'messages':kwargs['messages'],'request_model':kwargs['model'],'temperature':kwargs['temperature'],
               'max_tokens':kwargs['max_tokens'],'tool_schemas_sha256':hashlib.sha256(json.dumps(kwargs.get('tools'),sort_keys=True).encode()).hexdigest()}
            t=time.monotonic()
            try:
                response=original(**kwargs)
                e.update(response=response.model_dump(),status='success')
                return response
            except Exception as exc:
                e.update(status='error',error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                e['elapsed_seconds']=time.monotonic()-t;requests.append(e)
        teacher.client.chat.completions.create=capture
        row={**case,'teacher':cfg.teacher.model,'prompt_version':PROMPT_VERSION,'mock_version':MOCK_VERSION,
             'mock_catalog_version':CATALOG_VERSION,'mock_job_ids':plan['mock_job_ids'],'tools':SCHEMAS,
             'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'purpose':'diagnostic_only',
             'training_eligible':False,'case_attempt':experiment_attempt,'workflow_version':WORKFLOW_VERSION}
        try:
            row['messages']=tool_trace(teacher,cfg,source,case['question'],mock_job_ids=plan['mock_job_ids'],job_status_workflow=True)
            row['verification']=check_trace(row)
            calls=[c for m in row['messages'] if m['role']=='assistant' for c in message_calls(m)]
            decision='call' if calls else 'no_call'
            row['deterministic_case_check']={'decision':decision,'complete':bool(final_answer(row).strip()),
                'expected_decision':'call' if case['mode']=='call' else 'no_call',
                'exact_requested_call':(len(calls)==1 and calls[0]=={'name':'job_status','arguments':{'job_id':case['expected_job_id']}}) if case['mode']=='call' else not calls}
            row['status']='complete' if row['deterministic_case_check']['complete'] else 'incomplete'
        except Exception as exc:
            row.update(status='error',error=f'{type(exc).__name__}: {exc}')
        row['request_attempts']=attempt
        records.append(row)
        return row
    with ThreadPoolExecutor(max_workers=cfg.teacher.concurrency) as pool:
        rows=list(pool.map(one,plan['cases']))
    result={'job_id':os.environ['SLURM_JOB_ID'],'planned':12,'attempted':len(rows),'case_retries':0,
            'experiment_attempt':experiment_attempt,'replayed_cases':len(rows) if experiment_attempt>1 else 0,
            'completed':sum(r['status']=='complete' for r in rows),'errors':sum(r['status']=='error' for r in rows),
            'pipeline_pass':sum(r.get('verification',{}).get('passed',False) for r in rows),
            'planned_modes':dict(Counter(r['mode'] for r in rows)),
            'request_attempts':sum(r['request_attempts'] for r in rows),'elapsed_seconds':time.monotonic()-started,
            'training_trace_policy_version':TRAINING_TRACE_POLICY_VERSION,'quality_review':'PENDING','new_training_eligible':0}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--port',type=int);a=p.parse_args()
    if a.port:run(a.root.resolve(),a.port)
    else:
        cfg,plan,source,expected=preflight(a.root.resolve())
        print(json.dumps({'status':'PASS','planned':12,'modes':plan['modes'],'gold_guard':'PASS','gold_replay':'PASS',
                          'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'module_origin':distillkit.generate.__file__}))
