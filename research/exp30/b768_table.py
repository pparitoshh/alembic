"""Leaderboard tables: original, and with an equal 768-token answer budget (each model's *_b768 copy, answers cut at
768 tokens and re-judged, replaces the model where it exists; models within the budget are unchanged)."""
import json
D = 'runs/leaderboard'
R = {n: json.load(open(f'{D}/{f}'))['arms'] for n, f in [('gpt-oss', 'RESULT.json'), ('gemma', 'RESULT_gemma.json'),
                                                        ('qwen235', 'RESULT_qwen.json'), ('MAJORITY', 'RESULT_majority.json')]}
arms = sorted({k.split('/', 1)[1] for k in R['MAJORITY']})
def m(r, a):
    v = [r.get(f'{l}/{a}', {}).get('win_rate') for l in ('v3', 'v2', 'd1')]
    return v, (sum(v) / 3 if all(x is not None for x in v) else None)
def table(title, pick):
    print(title)
    print(f'{"model":20s} ' + ' | '.join(f'{n:>8s}' for n in R) + ' | majority v3 / v2 / d1')
    for a in sorted(pick, key=lambda a: -(m(R['MAJORITY'], a)[1] or 0)):
        means = [m(R[n], a)[1] for n in R]; v = m(R['MAJORITY'], a)[0]
        print(f'{a:20s} ' + ' | '.join('    -   ' if x is None else f'{x:8.3f}' for x in means) + ' | ' + ' / '.join('-' if x is None else f'{x:.3f}' for x in v))
    print()
table('ORIGINAL (no answer-length cap)', [a for a in arms if not a.endswith('_b768')])
table('EQUAL 768-TOKEN ANSWER BUDGET (answers over budget cut at 768 tokens, re-judged)',
      [a for a in arms if not a.endswith('_b768') and a + '_b768' not in arms] + [a for a in arms if a.endswith('_b768')])
