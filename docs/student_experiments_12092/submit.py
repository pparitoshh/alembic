"""Idempotently submit five additional runs and all six dependent evaluations."""
from pathlib import Path
import json,subprocess
HERE=Path(__file__).resolve().parent
p=HERE/'manifest.json'
m=json.loads(p.read_text())
def save():
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2)+'\n');tmp.replace(p)
def submit(*args):
    return subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]
for v in m['variants']:
    if not v.get('train_job_id'):
        v['train_job_id']=submit('--job-name=dk-'+v['name'],'--nice=50',str(HERE/'train.sbatch'),'--seed',str(v['seed']),'--learning-rate',str(v['learning_rate']),'--output',v['output']);save()
    if not v.get('eval_job_id'):
        v['eval_job_id']=submit('--job-name=dk-eval-'+v['name'],'--kill-on-invalid-dep=yes','--dependency=afterok:'+v['train_job_id'],str(HERE/'evaluate.sbatch'),v['name']);save()
if not m.get('aggregate_job_id'):
    m['aggregate_job_id']=submit('--job-name=dk-compare','--dependency=afterany:'+':'.join(v['eval_job_id'] for v in m['variants']),str(HERE/'aggregate.sbatch'));save()
m['status']='SUBMITTED';save()
print(json.dumps({'variants':m['variants'],'aggregate_job_id':m['aggregate_job_id']},indent=2))
