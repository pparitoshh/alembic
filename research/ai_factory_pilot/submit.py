"""Idempotent submission of the source-gated experiment DAG on Leonardo."""
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

repo = Path(os.environ['REPO']).resolve()
os.chdir(repo)
root = repo / 'runs/ai_factory_pilot_20261013'
receipt = root / 'submission.json'
smoke = sys.argv[1]
assert smoke.isdigit()
state = json.loads(receipt.read_text()) if receipt.exists() else {
    'created_at':datetime.now(timezone.utc).isoformat(), 'repo':str(repo),
    'smoke_job_id':smoke, 'jobs':{}, 'training_started':False,
    'huggingface_owner':'VaibhavKKM', 'model_published':False,
    'scope':'48 admitted LUMI technical pages; 256 proposed candidates, not a final dataset',
    'all_archive_trained':False}
assert state['smoke_job_id']==smoke, 'Existing DAG uses another smoke job'
logs=repo.parent/'logs'; logs.mkdir(exist_ok=True)

def submit(name,script,dependency):
    if name in state['jobs']:
        return state['jobs'][name]['job_id']
    command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--dependency='+dependency,
             '--output='+str(logs/(name+'-%A_%a.out')), str(repo/'research/ai_factory_pilot'/script)]
    subprocess.run([*command[:1], '--test-only', *command[2:]],check=True)
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    job=result.stdout.strip().split(';')[0]
    assert job.isdigit(), result.stdout
    state['jobs'][name]={'job_id':job,'dependency':dependency,'script':script}
    temporary=receipt.with_suffix('.json.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(receipt)
    print(name,job,dependency,flush=True)
    return job

data=submit('data','run.sbatch','afterok:'+smoke)
rag=submit('retrieval','retrieval.sbatch','afterok:'+smoke)
train=submit('train','train.sbatch','afterok:'+data)
submit('report','collect.sbatch','afterok:'+train+',afterany:'+rag)
print(json.dumps(state,indent=2))
