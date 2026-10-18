"""Report valid-flag coverage and conservative, non-submitting Slurm feasibility checks."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from distillkit.config import load_config
from distillkit.checks import check_answer, load_flags

cfg=load_config(sys.argv[1]);root=cfg.run_dir
out=root/'mentor_followup';out.mkdir(exist_ok=True)
flags=load_flags(cfg.verify.flag_list)
report={'metric_note':'Existing checker covers long options in #SBATCH directives and fenced sbatch/srun/salloc lines against a shared allowlist. It does not cover squeue/sacct/scontrol, short flags, option values or all inline prose; no detected error is not proof of correctness.',
        'scheduler_note':'Only complete Leonardo-specific batch scripts are tested as written with sbatch --test-only; no job is submitted. Other sites, placeholders, snippets and non-script answers are not scheduler passes.',
        'slices':{},'scheduler_checks':[]}
cache={}
for seed in [42,43,44]:
    for label in ['v2','v3']:
        rows=[json.loads(x) for x in (root/f'seed_{seed}/evaluation_{label}/eval_outputs.jsonl').read_text().splitlines()]
        metrics={}
        for model in ['base','student']:
            checks=[check_answer(r['answer_'+model],flags) for r in rows]
            containing=[c for c in checks if c['n_flags']>0]
            metrics[model]={'answers':len(rows),'answers_containing_flags':len(containing),
              'share_answers_without_detected_invalid_flags':sum(not c['bad_flags'] for c in checks)/len(checks),
              'share_flag_containing_answers_without_detected_invalid_flags':sum(not c['bad_flags'] for c in containing)/len(containing) if containing else None,
              'static_check_pass_rate':sum(c['passed'] for c in checks)/len(checks)}
            for row in rows:
                blocks=re.findall(r'```(?:bash|sh|shell)?\s*\n(.*?)```',row['answer_'+model],re.S)
                for i,script in enumerate(blocks):
                    if '#SBATCH' not in script:continue
                    entry={'seed':seed,'slice':label,'question_id':row['id'],'model':model,'block':i,'script_sha256':hashlib.sha256(script.encode()).hexdigest()}
                    if not re.search(r'\bLeonardo\b',row['question'],re.I):entry['status']='not_tested_unknown_or_different_cluster'
                    elif not script.startswith('#!'):entry['status']='not_tested_incomplete_script'
                    elif re.search(r'<[^>]+>|YOUR_|REPLACE_|TODO',script):entry['status']='not_tested_unbound_placeholders'
                    elif not shutil.which('sbatch'):entry['status']='not_tested_no_slurm'
                    else:
                        key=entry['script_sha256']
                        if key not in cache:
                            with tempfile.TemporaryDirectory(prefix='sbatch-audit-',dir=out) as tmp:
                                path=Path(tmp)/'candidate.sh';path.write_text(script+'\n')
                                try:
                                    result=subprocess.run(['sbatch','--test-only',str(path)],capture_output=True,text=True,timeout=15)
                                    cache[key]={'status':'scheduler_pass' if result.returncode==0 else 'scheduler_rejected',
                                                'returncode':result.returncode,'output':(result.stdout+result.stderr)[-5000:]}
                                except subprocess.TimeoutExpired:cache[key]={'status':'not_tested_scheduler_timeout'}
                        entry.update(cache[key])
                    report['scheduler_checks'].append(entry)
        report['slices'][f'{seed}/{label}']=metrics
eligible=[x for x in report['scheduler_checks'] if x['status'] in ['scheduler_pass','scheduler_rejected']]
report['scheduler_tested_scripts']=len(eligible)
report['scheduler_pass_rate']=sum(x['status']=='scheduler_pass' for x in eligible)/len(eligible) if eligible else None
(out/'ABSOLUTE_METRICS.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'slices':report['slices'],'scheduler_tested_scripts':len(eligible),'scheduler_pass_rate':report['scheduler_pass_rate']},indent=2))
