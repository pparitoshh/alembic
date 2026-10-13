"""Aggregate a fixed two-configuration, three-seed development experiment."""
import json
from pathlib import Path
from statistics import mean,stdev
HERE=Path(__file__).resolve().parent

def distribution(values):
    assert len(values)==3
    return {'n':3,'mean':mean(values),'sample_std':stdev(values),'values':values}

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    available={};missing=[];training_hashes=set();fixed_configs=set()
    for v in manifest['variants']:
        out=HERE/'evaluation'/v['name']
        evidence=HERE.parent/v['output']/'training_evidence.json'
        if not evidence.exists():missing.append(v['name']+': missing training');continue
        t=json.loads(evidence.read_text())
        if t.get('status')!='PASS':missing.append(v['name']+': training not PASS');continue
        assert t['source_release_sha256']==manifest['dataset_sha256']
        assert t['seed_before_model_init']==v['seed']
        assert t['training_config']['learning_rate']==v['learning_rate']
        training_hashes.add(t['training_code_sha256'])
        fixed_configs.add(json.dumps({k:value for k,value in t['training_config'].items() if k not in ('seed','learning_rate')},sort_keys=True))
        scores={}
        for judge in ['oss','gemma']:
            status=out/f'judge_status_{judge}.json'
            if not status.exists() or json.loads(status.read_text())['status']!='PASS':
                missing.append(v['name']+': '+judge+' evaluation incomplete');continue
            summary=json.loads((out/f'eval_summary_{judge}.json').read_text())
            scores[judge]=summary
        if len(scores)==2:available[v['name']]=scores
    assert len(fixed_configs)<=1,'More than learning rate and seed changed'
    assert len(training_hashes)<=1,'Training implementation differs across configurations'
    report={'status':'COMPLETE' if len(available)==6 else 'INCOMPLETE','scope':manifest['evaluation_scope'],'dataset_sha256':manifest['dataset_sha256'],'missing':missing,'configurations':{},'paired_learning_rate_differences':{},'limitations':['Three training seeds; spread is not a statistical significance claim.','Development evaluation used for configuration comparison; final holdout not evaluated.','Bash syntax/flag checks and isolated tool mocks do not establish compiled CUDA/MPI correctness or multi-turn completion.','No inference about semantic contamination from exact checks alone.']}
    if len(available)==6:
        for tag in ['lr1e4','lr5e5']:
            group={}
            for judge in ['oss','gemma']:
                group[judge]={'student_vs_base_win_rate':distribution([available[f'{tag}-seed{s}'][judge]['student_vs_base_win_rate'] for s in [42,43,44]])}
            for metric in ['check_pass_rate','tool_first_response_pass_rate','tool_decision_acc','tool_ast_acc']:
                group[metric]={model:distribution([available[f'{tag}-seed{s}']['oss'][model][metric] for s in [42,43,44]]) for model in ['base','student']}
            report['configurations'][tag]=group
        for judge in ['oss','gemma']:
            report['paired_learning_rate_differences'][judge]=distribution([available[f'lr5e5-seed{s}'][judge]['student_vs_base_win_rate']-available[f'lr1e4-seed{s}'][judge]['student_vs_base_win_rate'] for s in [42,43,44]])
    (HERE/'comparison_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
