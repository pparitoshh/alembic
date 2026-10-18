"""Retry only invalid judgments, preserving frozen answers and valid decisions."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

KEY = 'rag_vs_base'
VALID = {'A', 'B', 'T'}


def score_rows(rows):
    invalid = sum(v not in VALID for r in rows for v in r['judge'][KEY])
    if invalid or not rows:
        return None, invalid, None
    values = [({'A': 1., 'T': .5, 'B': 0.}[r['judge'][KEY][0]] +
               {'B': 1., 'T': .5, 'A': 0.}[r['judge'][KEY][1]]) / 2 for r in rows]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(5000))
    return sum(values) / len(values), 0, [boot[125], boot[4874]]


def repair_row(row, label, retry, attempts, save):
    result = deepcopy(row)
    for position, original in enumerate(row['judge'][KEY]):
        if original in VALID:
            continue
        prior = [a for a in attempts if (a['slice'], a['id'], a['position']) == (label, row['id'], position)]
        if any(a['verdict'] in VALID for a in prior):
            chosen = next(a for a in prior if a['verdict'] in VALID)
        else:
            chosen = None
            # One retry at the original budget, then one explicitly recorded larger budget.
            for number, budget in enumerate([4096, 8192], 1):
                if any(a['attempt'] == number for a in prior):
                    continue
                a, b = row['answer_rag'], row['answer_base']
                if position == 1:
                    a, b = b, a
                verdict, raw = retry(row, a, b, budget)
                chosen = dict(slice=label, id=row['id'], position=position, attempt=number,
                              max_tokens=budget, verdict=verdict, raw=raw)
                attempts.append(chosen)
                save(attempts)
                if verdict in VALID:
                    break
        if chosen is not None:
            result['judge'][KEY][position] = chosen['verdict']
            result['judge_raw'][KEY][position] = chosen['raw']
    return result


def main():
    from distillkit.config import load_config
    from distillkit.teacher import Teacher
    from distillkit.evaluate import judge
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output:
        raise ValueError('Use a separate output directory; frozen results must remain unchanged')
    output.mkdir(parents=True, exist_ok=True)
    names = ['benchmark_binding.json', 'BASE_RAG_RESULT.json'] + [
        f'{kind}_{label}.jsonl' for kind in ['answers', 'judgments'] for label in ['v2', 'v3']]
    binding = {'source': str(source), 'sha256': {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names},
        'repair_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'retry_policy_max_tokens': [4096, 8192], 'valid_judgments_preserved': True}
    stamp = output / 'repair_binding.json'
    if stamp.exists() and json.loads(stamp.read_text()) != binding:
        raise ValueError('Repair inputs changed; use a new output directory')
    stamp.write_text(json.dumps(binding, indent=2))
    cache = output / 'retry_attempts.json'
    attempts = json.loads(cache.read_text()) if cache.exists() else []

    def save(value):
        temporary = cache.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(cache)

    cfg = load_config(args.config)
    cfg.judge.model = 'openai/gpt-oss-20b'
    cfg.judge.base_url = args.endpoint
    cfg.judge.extra_body = {'reasoning_effort': 'medium'}

    def retry(row, a, b, budget):
        cfg.judge.max_tokens = budget
        return judge(Teacher(cfg.judge), row, a, b)

    report = json.loads((source / 'BASE_RAG_RESULT.json').read_text())
    for label, expected in [('v2', 66), ('v3', 200)]:
        rows = [json.loads(line) for line in (source / f'judgments_{label}.jsonl').read_text().splitlines()]
        answers = {r['id']: r for r in (json.loads(line) for line in (source / f'answers_{label}.jsonl').read_text().splitlines())}
        if len(rows) != expected or len({r['id'] for r in rows}) != expected or len(answers) != expected:
            raise ValueError('Incomplete or duplicate inputs: ' + label)
        for row in rows:
            if len(row['judge'][KEY]) != 2 or len(row['judge_raw'][KEY]) != 2:
                raise ValueError('Expected two counterbalanced judgments')
            for k, v in answers[row['id']].items():
                if row[k] != v:
                    raise ValueError('Answer input changed: ' + row['id'])
        resolved = [repair_row(row, label, retry, attempts, save) for row in rows]
        rate, invalid, interval = score_rows(resolved)
        report['slices'][label].update(win_rate=rate, invalid_judge_verdicts=invalid)
        report['slices'][label].pop('question_bootstrap_95_percent', None)
        if interval is not None:
            report['slices'][label]['question_bootstrap_95_percent'] = interval
        (output / f'judgments_{label}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in resolved))
    report['repair'] = {'original_result': str(source / 'BASE_RAG_RESULT.json'),
                        'original_files_unchanged': True, 'attempt_count': len(attempts),
                        'larger_judge_budget_used': any(a['max_tokens'] > 4096 for a in attempts),
                        'policy': 'Retry invalid judgments only: 4096 then 8192 tokens; preserve valid judgments.'}
    (output / 'BASE_RAG_RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if any(s['invalid_judge_verdicts'] for s in report['slices'].values()):
        raise SystemExit('Still incomplete after bounded retries; no aggregate score reported')


if __name__ == '__main__':
    main()
