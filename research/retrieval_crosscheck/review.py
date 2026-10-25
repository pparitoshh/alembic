"""Independent second-family review of frozen RAG/base answer pairs."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sys


def load_pairs(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('Incomplete or duplicated question set')
    for row in rows:
        verdicts = row['judge']['rag_vs_base']
        if len(verdicts) != 2 or any(v not in ['A', 'B', 'T'] for v in verdicts):
            raise ValueError('First-judge results must be complete')
        if not row['answer_base'].strip() or not row['answer_rag'].strip():
            raise ValueError('Empty original answer')
    return rows


def summarize(rows):
    swap = {'A': 'B', 'B': 'A', 'T': 'T'}
    invalid = sum(v not in swap for r in rows for v in r['judge']['rag_vs_base'])
    result = {'questions': len(rows), 'invalid_verdicts': invalid,
              'status': 'incomplete' if invalid else 'complete',
              'ordered_verdict_agreement': None, 'gemma_win_rate': None,
              'gemma_position_consistency': None, 'question_bootstrap_95_percent': None}
    if not rows or invalid:
        return result
    scores = [({'A': 1., 'T': .5, 'B': 0.}[r['judge']['rag_vs_base'][0]] +
               {'B': 1., 'T': .5, 'A': 0.}[r['judge']['rag_vs_base'][1]]) / 2 for r in rows]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(scores, k=len(scores))) / len(scores) for _ in range(5000))
    result.update(gemma_win_rate=sum(scores)/len(scores), question_bootstrap_95_percent=[boot[125], boot[4874]],
                  ordered_verdict_agreement=sum(a == b for r in rows for a, b in zip(r['gptoss_verdicts'], r['judge']['rag_vs_base'])) / (2 * len(rows)),
                  gemma_position_consistency=sum(r['judge']['rag_vs_base'][1] == swap[r['judge']['rag_vs_base'][0]] for r in rows) / len(rows))
    return result


def main():
    from distillkit.config import load_config
    from distillkit.evaluate import win_rate
    from distillkit.teacher import Teacher
    cfg = load_config(sys.argv[1])
    cfg.judge.base_url = sys.argv[2]
    assert cfg.judge.model == 'google/gemma-4-26B-A4B-it'
    out = cfg.run_dir
    out.mkdir(exist_ok=True)
    source = Path('runs/retrieval_2048_20261014/retrieval_repaired')
    files = {label: source/('judgments_'+label+'.jsonl') for label in ['v2', 'v3']}
    judge_config = cfg.judge.model_dump(mode='json')
    judge_config.pop('base_url', None)
    binding = {'judge': judge_config, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'source_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}}
    stamp = out/'binding.json'
    if stamp.exists():
        assert json.loads(stamp.read_text()) == binding, 'Input answers or reviewer changed'
    else:
        stamp.write_text(json.dumps(binding, indent=2))
    judge = Teacher(cfg.judge)
    report = {'purpose': 'Independent Gemma preference and agreement on actual frozen RAG/base answers',
              'first_judge': 'openai/gpt-oss-20b', 'second_judge': cfg.judge.model,
              'agreement_is_not_truth': True, 'first_judge_scores_unchanged': True,
              'human_labels_collected': 0, 'release_approved': False, 'slices': {}}
    pool = []
    for label, file in files.items():
        original = load_pairs(file, 66 if label == 'v2' else 200)
        pool.extend(original)
        cache = out/('gemma_'+label+'.jsonl')
        cached = [json.loads(x) for x in cache.read_text().splitlines()] if cache.exists() else []
        prior = {r['id']: r for r in cached}
        assert len(prior) == len(cached), 'Duplicate cached result'
        by_id = {r['id']: r for r in original}
        assert set(prior) <= set(by_id), 'Foreign cached result'
        for rid, row in prior.items():
            for field in ['question', 'reference', 'answer_base', 'answer_rag']:
                assert row[field] == by_id[rid][field], 'Cached answer binding mismatch'
            assert row['gptoss_verdicts'] == by_id[rid]['judge']['rag_vs_base']
        todo = [copy.deepcopy(r) for r in original if r['id'] not in prior]
        with cache.open('a') as handle:
            for offset in range(0, len(todo), 8):
                batch = todo[offset:offset+8]
                for row in batch:
                    row['gptoss_verdicts'] = list(row['judge']['rag_vs_base'])
                win_rate(judge, batch, 'rag', 'base')
                for row in batch:
                    handle.write(json.dumps(row)+'\n')
                    handle.flush()
                    prior[row['id']] = row
                print(label, 'crosschecked', len(prior), '/', len(original), flush=True)
        report['slices'][label] = summarize([prior[r['id']] for r in original])
        (out/'RAG_SECOND_JUDGE.json').write_text(json.dumps(report, indent=2)+'\n')
    # Optional human review evidence, never pretend these blank labels were supplied.
    rng = random.Random(37)
    packet, key = [], []
    for index, row in enumerate(rng.sample(pool, 30)):
        names = ['rag', 'base']
        rng.shuffle(names)
        rid = 'rag-real-%02d' % (index+1)
        packet.append({'review_id': rid, 'question': row['question'], 'reference': row['reference'],
                       'answer_A': row['answer_'+names[0]], 'answer_B': row['answer_'+names[1]],
                       'human_verdict': None, 'human_notes': None})
        key.append({'review_id': rid, 'question_id': row['id'], 'answer_A_model': names[0], 'answer_B_model': names[1]})
    (out/'human_review_packet_30.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in packet))
    (out/'human_review_key_30.json').write_text(json.dumps(key, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if any(s['invalid_verdicts'] for s in report['slices'].values()):
        raise SystemExit('Incomplete Gemma verdicts; do not report a complete score')


if __name__ == '__main__':
    main()
