"""Development-only, hash-bound base/student comparison with calibrated judges."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CODE = HERE/'code'
sys.path.insert(0,str(CODE/'src'))

def sha(p):return sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def rows(p):return [json.loads(x) for x in p.open()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['answers','judge','preflight'])
    ap.add_argument('variant')
    ap.add_argument('--judge',choices=['oss','gemma'])
    ap.add_argument('--port',type=int)
    a=ap.parse_args()
    manifest=json.loads((HERE/'manifest.json').read_text())
    v=next(x for x in manifest['variants']+[manifest['historical_reference']] if x['name']==a.variant)
    expected_rows=v.get('dataset_rows',manifest['dataset_rows'])
    expected_hash=v.get('dataset_sha256',manifest['dataset_sha256'])
    for name,digest in manifest['eval_hashes'].items():assert sha(CODE/name)==digest,name
    train_root=BASE/v['output']
    out=HERE/'evaluation'/a.variant
    out.mkdir(parents=True,exist_ok=True)
    from distillkit.config import Config
    from distillkit import evaluate,calibrate
    from distillkit.tools import set_valid_flags
    from distillkit.toolcheck import score_tool_item
    from distillkit.checks import check_answer,load_flags
    raw=json.loads((BASE/'interim-training-1100-v1/config.json').read_text())
    raw['run_dir']=str(out)
    raw['student']['model']='/leonardo_work/EUHPC_D30_031/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554'
    raw['eval'].update(file=str(CODE/'data/eval/dataset_v1/development_normal.jsonl'),tool_file=str(CODE/'data/eval/dataset_v1/development_tools.jsonl'),calibration_file=str(CODE/'data/eval/judge_calibration.jsonl'),tool_protocol='tool-first-response-v2',tool_application_policy='hpc-tools-clarify-first-v1',max_new_tokens=512)
    raw['train']['seed']=v['seed']
    raw['train']['learning_rate']=v['learning_rate']
    if a.judge:
        raw['judge'].update(model='openai/gpt-oss-20b' if a.judge=='oss' else 'google/gemma-4-26B-A4B-it',base_url=f'http://127.0.0.1:{a.port}/v1',concurrency=4,max_tokens=4096,extra_body={'reasoning_effort':'medium'} if a.judge=='oss' else {'chat_template_kwargs':{'enable_thinking':False}})
        raw['eval']['tag']=a.judge
    cfg=Config.model_validate(raw)
    normal,tools=evaluate._eval_rows(cfg)
    assert (len(normal),len(tools))==(106,23)
    if a.phase=='preflight':
        print('PASS evaluation config and frozen input hashes');return
    assert os.environ.get('SLURM_JOB_ID')
    evidence=json.loads((train_root/'training_evidence.json').read_text())
    assert evidence['status']=='PASS' and evidence['records']==expected_rows
    assert evidence['source_release_sha256']==expected_hash
    if v.get('controlled_seed',True):
        assert evidence['seed_before_model_init']==v['seed']
        assert evidence['training_config']['learning_rate']==v['learning_rate']
    adapter=train_root/'run/adapter'
    assert sha(adapter/'adapter_model.safetensors')==evidence['adapter_sha256']
    link=out/'adapter'
    if not link.exists():link.symlink_to(adapter,target_is_directory=True)
    assert link.resolve()==adapter.resolve()
    save(out/f'effective_config_{a.phase}_{a.judge or "none"}.json',raw)
    if a.phase=='answers':
        answers=evaluate.run_answers(cfg)
        assert set(answers)=={'base','student'} and all(len(x)==129 for x in answers.values())
        flags=load_flags(cfg.verify.flag_list);set_valid_flags(flags)
        static={}
        for name,outputs in answers.items():
            checks=[check_answer(x,flags) for x in outputs[:106]]
            tool_scores=[score_tool_item(text,row) for text,row in zip(outputs[106:],tools)]
            static[name]={'check_pass_rate':sum(x['passed'] for x in checks)/106,'answers_with_bad_flags':sum(bool(x['bad_flags']) for x in checks),**evaluate.tool_summary(tool_scores,tools)}
        save(out/'static_scores.json',static)
        save(out/'answer_evidence.json',{'status':'ANSWERED','utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'training_job_id':evidence['job_id'],'adapter_sha256':evidence['adapter_sha256'],'dataset_sha256':expected_hash,'code_commit':manifest['code_commit'],'cache_sha256':sha(out/'eval_answers_v3.json'),'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},'limitations':['Development data, not a blind final benchmark.','Static checks and mock tool execution do not establish real cluster execution or multi-turn task completion.','Generation capped at 512 new tokens; the production path does not retain finish reasons.']})
    else:
        assert a.judge and a.port
        cache=out/'eval_answers_v3.json'
        answer_evidence=json.loads((out/'answer_evidence.json').read_text())
        assert sha(cache)==answer_evidence['cache_sha256']
        calibration=calibrate.run(cfg)
        gate=manifest['calibration_gate']
        passed=calibration['accuracy']>=gate['accuracy_min'] and calibration['position_consistency']>=gate['position_consistency_min'] and calibration['unparsed']==0
        if not passed:
            save(out/f'judge_status_{a.judge}.json',{'status':'BLOCKED_CALIBRATION','calibration':calibration});return
        summary=evaluate.run(cfg)
        assert sha(cache)==answer_evidence['cache_sha256'],'Cached answers changed during judging'
        save(out/f'judge_status_{a.judge}.json',{'status':'PASS' if summary['student_vs_base_win_rate'] is not None else 'INCOMPLETE_VERDICTS','calibration':calibration})

if __name__=='__main__':main()
