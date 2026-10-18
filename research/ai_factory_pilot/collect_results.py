"""Collect all planned seeds; never hide a failed seed or call win rate accuracy."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

root = Path(sys.argv[1])
seeds = [42, 43, 44]
reports = [json.loads((root / f'seed_{s}/PILOT_RESULT.json').read_text()) for s in seeds]
bindings = [json.loads((root / f'seed_{s}/training_binding.json').read_text()) for s in seeds]
assert len({b['data_sha256'] for b in bindings}) == 1, 'Seeds used different training data'
result = {'experiment':'AI Factory document-grounded BF16 LoRA pilot', 'seeds':seeds,
          'training_rows':bindings[0]['selected_rows'], 'data_sha256':bindings[0]['data_sha256'],
          'release_approved':False, 'huggingface_destination_owner':'VaibhavKKM',
          'slices':{}, 'limitations':['Win rate is not factual accuracy.',
          'Frozen v2/v3 are development comparisons, not an untouched final test.',
          'New AI Factory domain quality needs its own independently reviewed evaluation.',
          'Synthetic judge calibration does not replace real-pair review.',
          'Three seeds on the same small pilot do not establish full-domain quality.']}
for label in ['v2','v3']:
    by_seed = []
    for seed in seeds:
        rows=[json.loads(x) for x in (root/f'seed_{seed}/evaluation_{label}/eval_outputs.jsonl').read_text().splitlines()]
        scores={r['id']:({'A':1.,'T':.5,'B':0.}[r['judge']['student_vs_base'][0]]+
                          {'B':1.,'T':.5,'A':0.}[r['judge']['student_vs_base'][1]])/2 for r in rows}
        assert len(scores)==(66 if label=='v2' else 200)
        by_seed.append(scores)
    ids=sorted(by_seed[0]); assert all(sorted(s)==ids for s in by_seed)
    seed_means=[statistics.mean(s.values()) for s in by_seed]
    # Resample the same question IDs across seeds, not 3*N fictitiously independent questions.
    per_question=[statistics.mean(s[q] for s in by_seed) for q in ids]
    rng=random.Random(42)
    boot=sorted(statistics.mean(rng.choices(per_question,k=len(ids))) for _ in range(5000))
    result['slices'][label]={'questions':len(ids),'mean_win_rate':statistics.mean(seed_means),
        'seed_win_rates':dict(zip(map(str,seeds),seed_means)),
        'seed_standard_deviation':statistics.stdev(seed_means),
        'question_bootstrap_95_percent':[boot[125],boot[4874]],
        'per_seed_metrics':{str(s):r['slices'][label]['checks_and_tools'] for s,r in zip(seeds,reports)}}
rag=root/'retrieval/BASE_RAG_RESULT.json'
if rag.exists():
    result['separate_document_assisted_baseline']=json.loads(rag.read_text())
else:
    result['separate_document_assisted_baseline']={'status':'not_finished'}
(root/'MULTISEED_RESULT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
