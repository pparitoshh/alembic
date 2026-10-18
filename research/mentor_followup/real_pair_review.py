"""Second-family review of actual student/base answers, without replacing first-judge scores."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from distillkit.config import load_config
from distillkit.evaluate import win_rate
from distillkit.teacher import Teacher

cfg=load_config(sys.argv[1]); endpoint=sys.argv[2]
root=cfg.run_dir; out=root/'mentor_followup';out.mkdir(exist_ok=True)
cfg.judge.model='google/gemma-4-26B-A4B-it';cfg.judge.base_url=endpoint
cfg.judge.temperature=0;cfg.judge.max_tokens=8192
judge=Teacher(cfg.judge)
files={f'{seed}/{label}':root/f'seed_{seed}/evaluation_{label}/eval_outputs.jsonl'
       for seed in [42,43,44] for label in ['v2','v3']}
binding={'judge':cfg.judge.model_dump(mode='json'),
         'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'real_pairs_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}}
# Endpoint port is transport only and can change on a resumed Slurm allocation.
binding['judge'].pop('base_url',None)
stamp=out/'real_pair_binding.json'
if stamp.exists():
    assert json.loads(stamp.read_text())==binding,'Real pairs or reviewer changed'
else:stamp.write_text(json.dumps(binding,indent=2))
summary={'purpose':'gpt-oss vs Gemma agreement on real answers; agreement is not truth',
         'first_judge':'openai/gpt-oss-20b','second_judge':cfg.judge.model,'human_labels_collected':0,
         'comparisons':{},'release_approved':False}
for key,path in files.items():
    original=[json.loads(x) for x in path.read_text().splitlines()]
    expected=66 if key.endswith('v2') else 200
    assert len(original)==expected and len({r['id'] for r in original})==expected
    cache=out/('gemma_'+key.replace('/','_')+'.jsonl')
    prior={r['id']:r for r in (json.loads(x) for x in cache.read_text().splitlines())} if cache.exists() else {}
    todo=[copy.deepcopy(r) for r in original if r['id'] not in prior]
    with cache.open('a') as f:
        for offset in range(0,len(todo),16):
            batch=todo[offset:offset+16]
            for row in batch:
                row['gptoss_verdicts']=row['judge']['student_vs_base']
                assert all(v in ['A','B','T'] for v in row['gptoss_verdicts'])
            win_rate(judge,batch,'student','base')
            for row in batch:
                f.write(json.dumps(row)+'\n');f.flush();prior[row['id']]=row
            print(key,'crosschecked',len(prior),'/',expected,flush=True)
    rows=[prior[r['id']] for r in original]
    invalid=sum(v not in ['A','B','T'] for r in rows for v in r['judge']['student_vs_base'])
    valid=[r for r in rows if all(v in ['A','B','T'] for v in r['judge']['student_vs_base'])]
    agreement=sum(a==b for r in valid for a,b in zip(r['gptoss_verdicts'],r['judge']['student_vs_base']))
    swap={'A':'B','B':'A','T':'T'}
    gemma_scores=[({'A':1.,'T':.5,'B':0.}[r['judge']['student_vs_base'][0]]+
                   {'B':1.,'T':.5,'A':0.}[r['judge']['student_vs_base'][1]])/2 for r in valid]
    summary['comparisons'][key]={'questions':expected,'invalid_verdicts':invalid,
        'status':'complete' if not invalid else 'incomplete',
        'ordered_verdict_agreement':agreement/(2*len(valid)) if valid and not invalid else None,
        'gemma_win_rate':sum(gemma_scores)/len(gemma_scores) if valid and not invalid else None,
        'gemma_position_consistency':sum(r['judge']['student_vs_base'][1]==swap[r['judge']['student_vs_base'][0]] for r in valid)/len(valid) if valid and not invalid else None}
    (out/'REAL_PAIR_AGREEMENT.json').write_text(json.dumps(summary,indent=2))
# Randomized anonymous presentation for 30 actual seed-42 pairs. No fabricated human verdicts.
pool=[json.loads(x) for label in ['v2','v3'] for x in files['42/'+label].read_text().splitlines()]
rng=random.Random(37);sample=rng.sample(pool,30);packet=[];answer_key=[]
for index,row in enumerate(sample):
    names=['student','base'];rng.shuffle(names)
    packet.append({'review_id':f'real-{index+1:02d}','question':row['question'],'reference':row.get('reference'),
        'answer_A':row['answer_'+names[0]],'answer_B':row['answer_'+names[1]],
        'human_verdict':None,'human_notes':None})
    answer_key.append({'review_id':packet[-1]['review_id'],'question_id':row['id'],'seed':42,
                       'answer_A_model':names[0],'answer_B_model':names[1]})
(out/'human_review_packet_30.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in packet))
(out/'human_review_key_30.json').write_text(json.dumps(answer_key,indent=2))
print(json.dumps(summary,indent=2))
if any(r['invalid_verdicts'] for r in summary['comparisons'].values()):
    raise SystemExit('Incomplete second-judge results: no full agreement claim')
