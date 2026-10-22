"""Generate training samples in fixed units that can resume with a different GPU count."""
import json
import os
from pathlib import Path
import sys

from evidence import EvidenceError, rows
from supervisor import lock


UNIT_SIZE = 64


def unit_rows(data):
    records = list(rows(data).values())
    return [records[i:i + UNIT_SIZE] for i in range(0, len(records), UNIT_SIZE)]


def completed(path, expected):
    found = rows(path)
    if set(found) != set(expected):
        raise EvidenceError('Generation unit does not contain exactly its assigned questions: ' + str(path))
    for row in found.values():
        if not isinstance(row.get('candidates'), list) or any(not isinstance(c, str) or not c.strip() for c in row['candidates']):
            raise EvidenceError('Malformed generated candidates: ' + str(path))
    return found


def recover(path, expected):
    """Keep successful rows; only a torn final journal line may be discarded."""
    found = {}
    if path.exists():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError:
                if i != len(lines) - 1:
                    raise EvidenceError('Corrupt generation journal: ' + str(path))
                break
            if row['id'] not in expected or row['id'] in found:
                raise EvidenceError('Unexpected or repeated generation ID: ' + str(path))
            found[row['id']] = row
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in found.values()))
    return found


def main(worker, workers, endpoint, served, data, out, samples):
    from distillkit.config import load_config
    from distillkit.teacher import Teacher

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config('configs/rag65_seed42.json')
    cfg.teacher.model, cfg.teacher.base_url = served, endpoint
    cfg.teacher.temperature, cfg.teacher.top_logprobs, cfg.teacher.concurrency, cfg.teacher.max_tokens = .7, 0, 192, 760
    cfg.teacher.extra_body = {'chat_template_kwargs': {'enable_thinking': False}}
    client = Teacher(cfg.teacher)
    system = cfg.student.system_prompt + os.environ.get('AUTOPILOT_TRAINING_NOTE', '')
    sample_seed = int(os.environ['AUTOPILOT_SAMPLE_SEED'])

    def sample(item):
        row, k = item
        try:
            answer = client.complete([{'role': 'system', 'content': system}, row['messages'][0]],
                                     extra_body={'seed': sample_seed + k})
            text = answer.content if answer.finish_reason == 'stop' and (answer.content or '').strip() else None
            return row['id'], k, text, True
        except Exception as exc:
            print('transient sample failure', row['id'], k, type(exc).__name__, flush=True)
            return row['id'], k, None, False

    for number, unit in enumerate(unit_rows(data)):
        if number % workers != worker:
            continue
        expected = {r['id'] for r in unit}
        dest = out / f'cands_unit_{number:05d}.jsonl'
        with lock(dest.with_suffix('.lock')) as acquired:
            if not acquired:
                continue
            if dest.exists():
                completed(dest, expected)
                continue
            journal = dest.with_suffix('.partial')
            done = recover(journal, expected)
            todo = [r for r in unit if r['id'] not in done]
            got, failures = {}, set()
            for rid, k, text, ok in client.map(sample, [(r, k) for r in todo for k in range(samples)]):
                if not ok:
                    failures.add(rid)
                elif text is not None:
                    got.setdefault(rid, {})[k] = text
            with journal.open('a') as f:
                for row in todo:
                    rid = row['id']
                    if rid in failures:
                        continue
                    result = {'id': rid, 'teacher': served, 'sample_seed': sample_seed,
                              'candidates': [got.get(rid, {})[k] for k in sorted(got.get(rid, {}))]}
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    done[rid] = result
            if set(done) == expected:
                completed(journal, expected)
                journal.replace(dest)
                print('completed', dest.name, len(unit), flush=True)
            else:
                print('partial', dest.name, len(done), '/', len(expected), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:7], int(sys.argv[7]) if len(sys.argv) > 7 else 4)
