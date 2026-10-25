from copy import deepcopy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('repair', Path(__file__).with_name('repair_judgments.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_retry_preserves_valid_and_original_and_position():
    row = dict(id='q', answer_rag='rag', answer_base='base',
               judge={'rag_vs_base': ['A', None]}, judge_raw={'rag_vs_base': ['valid', '']})
    original = deepcopy(row)
    calls, attempts = [], []
    def retry(q, a, b, budget):
        calls.append((a, b, budget))
        return ('B', 'fixed') if budget == 8192 else (None, '')
    out = module.repair_row(row, 'v3', retry, attempts, lambda _: None)
    assert row == original
    assert out['judge']['rag_vs_base'] == ['A', 'B']
    assert out['judge_raw']['rag_vs_base'] == ['valid', 'fixed']
    assert calls == [('base', 'rag', 4096), ('base', 'rag', 8192)]
    assert module.repair_row(row, 'v3', retry, attempts, lambda _: None) == out
    assert len(calls) == 2


def test_invalid_never_becomes_tie_and_retries_bounded():
    row = dict(id='q', answer_rag='rag', answer_base='base',
               judge={'rag_vs_base': [None, 'T']}, judge_raw={'rag_vs_base': ['', 'tie']})
    attempts = []
    out = module.repair_row(row, 'v2', lambda *a: (None, ''), attempts, lambda _: None)
    assert module.score_rows([out]) == (None, 1, None)
    out2 = module.repair_row(row, 'v2', lambda *a: (_ for _ in ()).throw(AssertionError('unbounded retry')), attempts, lambda _: None)
    assert out2 == out and len(attempts) == 2


def test_counterbalanced_score():
    rows = [dict(judge={'rag_vs_base': values}) for values in [['A', 'B'], ['B', 'A'], ['T', 'T']]]
    rate, invalid, interval = module.score_rows(rows)
    assert rate == .5 and invalid == 0 and interval[0] <= rate <= interval[1]
