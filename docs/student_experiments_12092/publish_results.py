"""Publish finished experiment results, including negative/incomplete outcomes."""
import fcntl,hashlib,json,os,shutil,subprocess,tarfile,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
REMOTE='/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z'
REPO='pparitoshh/alembic'
TAG='student-comparison-12092-v1'
EXPECTED='0d26fd2ec8fa5fb5221d22aa6249f6d107201d9f4ae0b06da4734fd105f7fd9c'
os.environ['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
def run(*args,cwd=None,check=True,timeout=3600):
    return subprocess.run(args,cwd=cwd,check=check,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (HERE/'published.json').exists():return
    r=run('ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','leonardo','cat '+REMOTE+'/student-experiments-v1/comparison_report.json',check=False,timeout=45)
    if r.returncode:
        print('WAITING: comparison report unavailable',flush=True);return
    report=json.loads(r.stdout)
    assert report['status'] in ('COMPLETE','INCOMPLETE') and report['dataset_sha256']==EXPECTED
    evidence=HERE/'results';evidence.mkdir(exist_ok=True)
    (evidence/'comparison_report.json').write_text(r.stdout)
    run('rsync','-aq','--exclude','adapter','--include','*/','--include','answer_evidence.json','--include','static_scores.json','--include','eval_summary_*.json','--include','calibration_summary_*.json','--include','judge_status_*.json','--exclude','*','leonardo:'+REMOTE+'/student-experiments-v1/evaluation/',str(evidence/'evaluation')+'/')
    m=json.loads(run('ssh','leonardo','cat '+REMOTE+'/student-experiments-v1/manifest.json').stdout)
    completed=[];artifacts=HERE/'artifacts';artifacts.mkdir(exist_ok=True)
    for v in m['variants']:
        r=run('ssh','leonardo','cat '+REMOTE+'/'+v['output']+'/training_evidence.json',check=False)
        if r.returncode:continue
        t=json.loads(r.stdout)
        if t.get('status')!='PASS':continue
        assert t['source_release_sha256']==EXPECTED and t['seed_before_model_init']==v['seed']
        dst=artifacts/v['name'];dst.mkdir(exist_ok=True)
        run('rsync','-aq','leonardo:'+REMOTE+'/'+v['output']+'/run/adapter/',str(dst/'adapter')+'/')
        run('rsync','-q','leonardo:'+REMOTE+'/'+v['output']+'/config.json',str(dst/'config.json'))
        assert sha(dst/'adapter/adapter_model.safetensors')==t['adapter_sha256']
        assert sha(dst/'config.json')==t['config_sha256']
        (dst/'training_evidence.json').write_text(r.stdout)
        completed.append({'variant':v['name'],'adapter_sha256':t['adapter_sha256'],'training_job_id':t['job_id']})
    archive=HERE/'student-comparison-12092.tar.gz'
    if not archive.exists():
        temp=archive.with_suffix('.tmp')
        with tarfile.open(temp,'w:gz') as tar:
            tar.add(artifacts,arcname='students')
            tar.add(evidence,arcname='results')
        temp.replace(archive)
    bundle={'completed_variants':completed,'report_status':report['status'],'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'release':f'https://github.com/{REPO}/releases/tag/{TAG}'}
    sums=HERE/'SHA256SUMS';sums.write_text(bundle['archive_sha256']+'  '+archive.name+'\n')
    with tempfile.TemporaryDirectory(prefix='publish-',dir=HERE) as tmp:
        repo=Path(tmp)/'repo';run('git','clone','--depth','1','--branch','dev',f'https://github.com/{REPO}.git',str(repo))
        d=repo/'docs/student_experiments_12092';d.mkdir(exist_ok=True)
        shutil.copytree(evidence,d/'results',dirs_exist_ok=True)
        (d/'artifact_manifest.json').write_text(json.dumps(bundle,indent=2)+'\n')
        (d/'RESULTS.md').write_text(f'''# Student comparison results

Experiment status: **{report['status']}**. See [machine-readable report](results/comparison_report.json) for measured results, missing runs and limitations. No positive improvement is inferred from completion alone.

The artifact release contains {len(completed)} completed, hash-verified adapters and their training evidence: {bundle['release']} . Configuration comparisons use the same 12,092-row snapshot, two learning rates and paired seeds 42/43/44. Reports use development data; the final holdout remains separate. Incomplete or calibration-blocked results are disclosed rather than silently dropped.
''')
        quality=repo/'docs/QUALITY_REVIEW_STATUS.md'
        content=quality.read_text()
        marker='## Controlled 12,092-example student comparison'
        if marker not in content:quality.write_text(content+'\n'+marker+'\n\nSee [results and artifacts](student_experiments_12092/RESULTS.md). Status: **'+report['status']+'**. This is development evaluation, not a final held-out improvement claim.\n')
        run('git','add','docs/student_experiments_12092','docs/QUALITY_REVIEW_STATUS.md',cwd=repo)
        if run('git','diff','--cached','--quiet',cwd=repo,check=False).returncode:
            run('git','commit','-m','Publish controlled student comparison results and artifact manifest',cwd=repo)
            run('git','push','origin','HEAD:dev',cwd=repo)
        commit=run('git','rev-parse','HEAD',cwd=repo).stdout.strip()
        if run('gh','release','view',TAG,'--repo',REPO,check=False).returncode:
            run('gh','release','create',TAG,str(archive),str(sums),'--repo',REPO,'--target',commit,'--prerelease','--title','QDoRA seed comparison: 12,092 examples','--notes-file',str(d/'RESULTS.md'))
        else:run('gh','release','upload',TAG,str(archive),str(sums),'--repo',REPO,'--clobber')
        assets=json.loads(run('gh','release','view',TAG,'--repo',REPO,'--json','assets').stdout)['assets']
        assert {archive.name,'SHA256SUMS'}<={x['name'] for x in assets}
        (HERE/'published.json').write_text(json.dumps({'commit':commit,**bundle},indent=2)+'\n')
        print('PUBLISHED',commit,flush=True)
if __name__=='__main__':
    with (HERE/'publisher.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(0)
        main()
