import hashlib,json,os,subprocess
from pathlib import Path
repo=Path(os.environ['REPO']);os.chdir(repo)
root=repo/'runs/ai_factory_pilot_20261013'
receipt=root/'submission.json';state=json.loads(receipt.read_text())
if 'mentor_crosscheck' in state['jobs']:
    print(state['jobs']['mentor_crosscheck']);raise SystemExit(0)
folder=repo/'research/mentor_followup'
(root/'mentor_code_snapshot.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()},indent=2))
dep='afterok:'+state['jobs']['train']['job_id']
command=['sbatch','--parsable','--kill-on-invalid-dep=yes','--dependency='+dep,
         '--output='+str(repo.parent/'logs/mentor-%j.out'),str(folder/'crosscheck.sbatch')]
subprocess.run([command[0],'--test-only',*command[2:]],check=True)
r=subprocess.run(command,capture_output=True,text=True,check=True)
job=r.stdout.strip().split(';')[0];assert job.isdigit()
state['jobs']['mentor_crosscheck']={'job_id':job,'dependency':dep,'script':'research/mentor_followup/crosscheck.sbatch'}
temp=receipt.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(receipt)
print(json.dumps(state['jobs']['mentor_crosscheck'],indent=2))
