"""Report the pilot honestly, with question-level uncertainty and no automatic release."""
import json
from pathlib import Path
import random
import sys

root = Path(sys.argv[1])
report = {'experiment': 'AI Factory source-context BF16 LoRA pilot',
          'seed': int(sys.argv[2]), 'release_approved': False, 'all_factory_training_claim': False,
          'interpretation': 'Frozen v2/v3 measure HPC regression; they do not establish new AI Factory coverage. '
                            'A single pilot seed is not a multi-seed quality claim.', 'slices': {}}
for name in ['v2', 'v3']:
    folder = root / ('evaluation_' + name)
    summary = json.loads((folder / 'eval_summary.json').read_text())
    comparison = summary['judge_comparisons']['student_vs_base']
    if comparison['status'] != 'complete' or comparison['invalid_verdicts']:
        raise RuntimeError(name + ' has incomplete judgments; no aggregate result released')
    rows = [json.loads(line) for line in (folder / 'eval_outputs.jsonl').read_text().splitlines()]
    scores = [({'A': 1., 'T': .5, 'B': 0.}[r['judge']['student_vs_base'][0]] +
               {'B': 1., 'T': .5, 'A': 0.}[r['judge']['student_vs_base'][1]]) / 2 for r in rows]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(5000))
    report['slices'][name] = {'questions': len(scores), 'win_rate': sum(scores) / len(scores),
                             'question_bootstrap_95_percent': [boot[125], boot[4874]],
                             'checks_and_tools': {k: summary[k] for k in ['base', 'student']},
                             'judge': summary['judge']}
(root / 'PILOT_RESULT.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
