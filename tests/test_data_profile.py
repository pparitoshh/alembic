"""Suitability failures stop training before any GPU/model imports."""
import hashlib
import json

import pytest

from distillkit.config import Config, TrainingDataRequirementsCfg
from distillkit.data_profile import enforce_requirements, profile, records_sha256
from distillkit import train
from test_pipeline import _cfg_dict


def examples():
    rows = []
    for i, mode in enumerate(['call', 'ask', 'none'] + ['prose'] * 7):
        row = {'id': str(i), 'doc_id': f'doc{i}', 'mode': mode,
               'task': ['concept', 'howto', 'script', 'debug'][i % 4],
               'question': f'question {i}',
               'messages': [{'role': 'assistant', 'content': f'answer {i}'}]}
        if mode != 'prose':
            row['tools'] = [{'type': 'function', 'function': {'name': 'status'}}]
        if mode == 'call':
            row['messages'].insert(0, {'role': 'assistant', 'content': None,
                                      'tool_calls': [{'type': 'function', 'function': {'name': 'status', 'arguments': '{}'}}]})
        rows.append(row)
    return rows


def requirements(tmp_path, rows):
    evidence = tmp_path / 'relevance.json'
    evidence.write_text(json.dumps({'fixture': True, 'ids': [r['id'] for r in rows]}))
    review = tmp_path / 'domain_review.json'
    review.write_text(json.dumps({
        'version': 'training-domain-review-v1', 'records_sha256': records_sha256(rows),
        'decision': 'approved-for-hpc-training', 'approved_record_ids': [r['id'] for r in rows],
        'review_method': 'synthetic test fixture; not an actual relevance review',
        'evidence': {'path': evidence.name, 'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()},
    }))
    return TrainingDataRequirementsCfg(domain_review={
        'path': review, 'sha256': hashlib.sha256(review.read_bytes()).hexdigest()})


def test_call_ask_none_all_count_as_tool_decisions():
    result = profile(examples())
    assert result['tool_decision_rows'] == 3
    assert result['actual_tool_call_rows'] == 1
    assert result['tool_decision_fraction'] == 0.3
    assert result['mode_structure_errors'] == []


def test_bound_review_and_mix_pass(tmp_path):
    rows = examples()
    assert enforce_requirements(rows, requirements(tmp_path, rows))['status'] == 'PASS'


@pytest.mark.parametrize('tamper', ['data', 'review', 'evidence'])
def test_rejects_changed_inputs(tmp_path, tamper):
    rows = examples()
    policy = requirements(tmp_path, rows)
    if tamper == 'data':
        rows[-1]['messages'][-1]['content'] = 'changed answer'
    else:
        path = policy.domain_review.path if tamper == 'review' else tmp_path / 'relevance.json'
        path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='domain review:'):
        enforce_requirements(rows, policy)


def test_approved_data_still_fails_bad_mix(tmp_path):
    rows = [r for r in examples() if r['mode'] == 'prose']
    with pytest.raises(ValueError, match='tool-decision fraction'):
        enforce_requirements(rows, requirements(tmp_path, rows), evidence_dir=tmp_path)
    saved = json.loads((tmp_path / 'training_data_profile.json').read_text())
    assert saved['status'] == 'FAIL'


def test_mode_label_without_tools_does_not_pass(tmp_path):
    rows = examples()
    rows[1].pop('tools')
    with pytest.raises(ValueError, match='mode/structure'):
        enforce_requirements(rows, requirements(tmp_path, rows))


def test_missing_review_blocks_production_train_before_imports(tmp_path, monkeypatch):
    rows = examples()
    policy = requirements(tmp_path, rows)
    policy.domain_review.path.unlink()
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    cfg.train.data_requirements = policy
    cfg.run_dir.mkdir()
    (cfg.run_dir / 'verified.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    import builtins
    original = builtins.__import__
    def no_gpu(name, *args, **kwargs):
        assert name not in {'torch', 'transformers', 'trl'}, 'loaded training stack before admission'
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_gpu)
    with pytest.raises(ValueError, match='domain review:'):
        train.run(cfg)


def test_manifest_input_also_runs_suitability_gate(tmp_path, monkeypatch):
    from distillkit import training_inputs
    rows = examples()[3:]
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    cfg.train.input_manifest = {'placeholder': True}  # admission replaced by fixture below
    cfg.train.data_requirements = requirements(tmp_path, rows)
    monkeypatch.setattr(training_inputs, 'admitted_rows', lambda *a, **k: rows)
    with pytest.raises(ValueError, match='tool-decision fraction'):
        train.admitted_training_rows(cfg)


def test_absent_policy_preserves_historical_fingerprint(tmp_path):
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    old_train = cfg.train.model_dump(mode='json', exclude={'save_steps', 'save_total_limit', 'report_to', 'data_requirements'})
    expected = hashlib.sha256(json.dumps({'examples': [], 'student': cfg.student.model, 'train': old_train},
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert train.inputs_fingerprint(cfg, []) == expected
