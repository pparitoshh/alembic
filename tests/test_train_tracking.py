"""Validation split, kept checkpoint adapters and MLflow checkpoint tracking; no model, GPU or network."""
import json

import pytest

from distillkit import train, tracking


def rows_for(docs):
    return [{'id': f'{d}#{i}', 'doc_id': d} for d, n in docs.items() for i in range(n)]


def test_no_fraction_keeps_every_example_for_training():
    rows = rows_for({'a': 3})
    assert train.split_by_document(rows, rows, 0.0, 1) == (rows, [])


def test_validation_takes_whole_documents_until_the_fraction_is_reached():
    rows = rows_for({f'd{i}': 5 for i in range(20)})
    tr, val = train.split_by_document(rows, rows, 0.1, seed=42)
    val_docs, tr_docs = {r['doc_id'] for r in val}, {r['doc_id'] for r in tr}
    assert len(val) == 10 and len(tr) == 90
    assert not val_docs & tr_docs
    assert train.split_by_document(rows, rows, 0.1, seed=42) == (tr, val)
    assert train.split_by_document(rows, rows, 0.1, seed=7)[1] != val


def test_split_refuses_to_leave_no_training_data():
    rows = rows_for({'only': 4})
    with pytest.raises(ValueError, match='no training examples'):
        train.split_by_document(rows, rows, 0.4, seed=1)


def test_keep_adapter_copies_adapter_without_optimizer_state(tmp_path):
    ckpt = tmp_path / 'checkpoints' / 'checkpoint-100'
    ckpt.mkdir(parents=True)
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json', 'optimizer.pt',
                 'scheduler.pt', 'rng_state.pth', 'training_args.bin'):
        (ckpt / name).write_text(name)
    dst = train.keep_adapter(ckpt, tmp_path, 100)
    assert dst == tmp_path / 'adapters' / 'step_100' / 'adapter'
    assert sorted(f.name for f in dst.iterdir()) == ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json']


def summary(win):
    return {'base': {'check_pass_rate': 0.5, 'answers_with_bad_flags': 3},
            'student': {'check_pass_rate': 0.6, 'tool_first_response_pass_rate': 0.4, 'tool_protocol': 'v2'},
            'student_vs_base_win_rate': win, 'judge_unparsed': 0, 'judge': 'openai/gpt-oss-20b',
            'judge_comparisons': {'student_vs_base': {'status': 'complete'}}}


def write_summaries(run_dir, wins):
    for step, win in wins.items():
        d = run_dir / 'adapters' / f'step_{step}'
        d.mkdir(parents=True)
        (d / 'eval_summary.json').write_text(json.dumps(summary(win)))


def test_checkpoint_metrics_flatten_numbers_by_step(tmp_path):
    write_summaries(tmp_path, {100: 0.55, 1000: 0.7})
    metrics = tracking.checkpoint_metrics(tmp_path)
    assert sorted(metrics) == [100, 1000]
    assert metrics[1000] == {'dev/base/check_pass_rate': 0.5, 'dev/base/answers_with_bad_flags': 3.0,
                             'dev/student/check_pass_rate': 0.6, 'dev/student/tool_first_response_pass_rate': 0.4,
                             'dev/student_vs_base_win_rate': 0.7, 'dev/judge_unparsed': 0.0}


def test_incomplete_judging_is_not_logged_as_a_score(tmp_path):
    write_summaries(tmp_path, {100: None})
    assert 'dev/student_vs_base_win_rate' not in tracking.checkpoint_metrics(tmp_path)[100]


def test_track_logs_into_the_training_run_by_name(tmp_path, monkeypatch):
    mlflow = pytest.importorskip('mlflow')
    from distillkit.config import Config
    from test_pipeline import _cfg_dict
    monkeypatch.setenv('MLFLOW_TRACKING_URI', (tmp_path / 'mlruns').as_uri())
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE', 'true')  # as slurm/env.sh sets it
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    with mlflow.start_run(run_name='other'):
        pass
    with mlflow.start_run(run_name=str(cfg.run_dir)) as r:
        run_id = r.info.run_id
    write_summaries(cfg.run_dir, {100: 0.55, 200: 0.6})
    tracking.run(cfg)
    history = mlflow.MlflowClient().get_metric_history(run_id, 'dev/student_vs_base_win_rate')
    assert [(m.step, m.value) for m in history] == [(100, 0.55), (200, 0.6)]


def test_track_fails_without_a_training_run(tmp_path, monkeypatch):
    pytest.importorskip('mlflow')
    from distillkit.config import Config
    from test_pipeline import _cfg_dict
    monkeypatch.setenv('MLFLOW_TRACKING_URI', (tmp_path / 'mlruns').as_uri())
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE', 'true')  # as slurm/env.sh sets it
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    with pytest.raises(ValueError, match='no MLflow run named'):
        tracking.run(cfg)
