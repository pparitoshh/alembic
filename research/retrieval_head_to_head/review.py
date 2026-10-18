"""Compare two frozen retrieval trials directly, with a base-answer drift control.

No answers are regenerated. Both answer orders have equal question weight.
This is a development preference comparison, not factual accuracy or a release.
"""
import hashlib
import json
from pathlib import Path
import random
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pairs(old_path, new_path, expected, variant):
    old = [json.loads(l) for l in old_path.read_text().splitlines()]
    new = [json.loads(l) for l in new_path.read_text().splitlines()]
    if any(len(rows) != expected or len({r['id'] for r in rows}) != expected for rows in [old, new]):
        raise ValueError('Incomplete or duplicate frozen answers')
    old_by_id = {r['id']: r for r in old}
    if set(old_by_id) != {r['id'] for r in new}:
        raise ValueError('Different question IDs')
    pairs = []
    for n in new:
        o = old_by_id[n['id']]
        for field in ['question', 'reference']:
            if o[field] != n[field]:
                raise ValueError('Frozen question/reference mismatch: ' + n['id'])
        for r in [o, n]:
            if not r['answer_' + variant].strip():
                raise ValueError('Empty frozen answer')
        pairs.append({'id': n['id'], 'question': n['question'], 'reference': n['reference'],
                      'answer_old': o['answer_' + variant], 'answer_new': n['answer_' + variant],
                      'finish_reason_old': o['finish_reason_' + variant],
                      'finish_reason_new': n['finish_reason_' + variant]})
    return pairs


def summarize(rows):
    swap = {'A': 'B', 'B': 'A', 'T': 'T'}
    verdicts = [r['judge']['new_vs_old'] for r in rows]
    if any(len(v) != 2 for v in verdicts):
        raise ValueError('Expected exactly two ordered judgments per question')
    invalid = sum(v not in swap for pair in verdicts for v in pair)
    result = {'questions': len(rows), 'invalid_verdicts': invalid,
              'status': 'incomplete' if invalid or not rows else 'complete',
              'new_preference': None, 'paired_question_bootstrap_95_percent': None,
              'position_consistency': None,
              'identical_answers': sum(r['answer_new'] == r['answer_old'] for r in rows),
              'truncated_old': sum(r['finish_reason_old'] == 'length' for r in rows),
              'truncated_new': sum(r['finish_reason_new'] == 'length' for r in rows)}
    if invalid or not rows:
        return result
    scores = [({'A': 1., 'B': 0., 'T': .5}[v[0]] + {'B': 1., 'A': 0., 'T': .5}[v[1]]) / 2 for v in verdicts]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(5000))
    result.update(new_preference=sum(scores) / len(scores), paired_question_bootstrap_95_percent=[boot[125], boot[4874]],
                  position_consistency=sum(v[1] == swap[v[0]] for v in verdicts) / len(verdicts))
    return result


def main():
    from distillkit.config import load_config
    from distillkit.evaluate import win_rate
    from distillkit.teacher import Teacher
    cfg = load_config(sys.argv[1])
    cfg.judge.base_url = sys.argv[2]
    if cfg.judge.model != 'google/gemma-4-26B-A4B-it':
        raise ValueError('Expected independently configured Gemma reviewer')
    root = cfg.run_dir
    files = {name: root / 'inputs' / name for name in ['old_v2.jsonl', 'old_v3.jsonl', 'new_v2.jsonl', 'new_v3.jsonl']}
    jc = cfg.judge.model_dump(mode='json')
    jc.pop('base_url', None)
    binding = {'judge': jc, 'files': {n: sha(p) for n, p in files.items()}, 'code_sha256': sha(__file__)}
    stamp = root / 'review_binding.json'
    if stamp.exists() and json.loads(stamp.read_text()) != binding:
        raise ValueError('Frozen inputs or reviewer configuration changed')
    stamp.write_text(json.dumps(binding, indent=2) + '\n')
    judge = Teacher(cfg.judge)
    report = {'purpose': 'Direct frozen new-vs-old RAG answers plus separate base-answer drift control',
              'old_trial': 'retrieval_2048_20261014', 'new_trial': 'retrieval_option_aware_20261014',
              'reviewer': cfg.judge.model, 'answers_regenerated': False, 'release_approved': False,
              'accuracy_claim': False, 'comparisons': {},
              'limits': ['One serving realization per trial; not an estimate over repeated generation runs.',
                         'Same evaluation questions are used for development, not an untouched final test.',
                         'Base drift control is a separate diagnostic; do not subtract preferences as a causal effect.',
                         'Invalid judgments remain incomplete; do not choose a favorable retry.']}
    for variant in ['rag', 'base']:
        for label, expected in [('v2', 66), ('v3', 200)]:
            key = variant + '_' + label
            pairs = load_pairs(files['old_' + label + '.jsonl'], files['new_' + label + '.jsonl'], expected, variant)
            cache = root / (key + '.jsonl')
            cached = [json.loads(l) for l in cache.read_text().splitlines()] if cache.exists() else []
            by_id = {r['id']: r for r in pairs}
            prior = {r['id']: r for r in cached}
            if len(prior) != len(cached) or not set(prior) <= set(by_id):
                raise ValueError('Foreign or duplicated cached verdicts')
            for rid, row in prior.items():
                if any(row[k] != v for k, v in by_id[rid].items()):
                    raise ValueError('Cached source pair changed')
            todo = [r for r in pairs if r['id'] not in prior]
            with cache.open('a') as handle:
                for offset in range(0, len(todo), 8):
                    batch = todo[offset:offset+8]
                    win_rate(judge, batch, 'new', 'old')
                    for row in batch:
                        handle.write(json.dumps(row) + '\n')
                        handle.flush()
                        prior[row['id']] = row
                    print(key, 'judged', len(prior), '/', expected, flush=True)
            report['comparisons'][key] = summarize([prior[r['id']] for r in pairs])
            (root / 'HEAD_TO_HEAD_RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if any(s['status'] != 'complete' for s in report['comparisons'].values()):
        raise SystemExit('Incomplete ordered verdicts; no complete headline score')


if __name__ == '__main__':
    main()
