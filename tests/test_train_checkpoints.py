"""Checkpoint resume and length reporting for the student trainer; no model, GPU or cluster calls."""
import json

import pytest

from distillkit import train
from distillkit.config import Config
from distillkit.io import write_jsonl
from test_pipeline import _cfg_dict


class WordTokenizer:
    """One token per whitespace-separated word of the rendered chat."""
    def apply_chat_template(self, messages, tools=None, tokenize=False, **kw):
        assert tokenize is False
        return ' '.join(m['content'] for m in messages) + (' TOOLS' if tools else '')
    def __call__(self, text, **kw):
        return {'input_ids': text.split()}


def example(words, tools=None):
    return {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': ' '.join(['w'] * words)}],
            'tools': json.dumps(tools) if tools else None, 'chat_template_kwargs': {'enable_thinking': False}}


def test_token_lengths_render_messages_and_decoded_tools():
    lengths = train.token_lengths(WordTokenizer(), [example(3), example(3, tools=[{'type': 'function'}])])
    assert lengths == [4, 5]


def test_length_report_counts_examples_over_max_length():
    report = train.length_report([10, 20, 30, 40], max_length=25)
    assert 'median 30' in report and 'max 40' in report and '2 of 4 over max_length=25' in report


def test_fresh_run_stamps_inputs_and_starts_from_scratch(tmp_path):
    ckpt = tmp_path / 'checkpoints'
    assert train.resume_checkpoint(ckpt, 'abc') is None
    assert (ckpt / 'inputs.sha256').read_text().strip() == 'abc'


def test_resume_picks_newest_step_numerically(tmp_path):
    ckpt = tmp_path / 'checkpoints'
    train.resume_checkpoint(ckpt, 'abc')
    for step in (100, 900, 1000):
        (ckpt / f'checkpoint-{step}').mkdir()
    (ckpt / 'checkpoint-tmp').mkdir()
    assert train.resume_checkpoint(ckpt, 'abc') == str(ckpt / 'checkpoint-1000')


@pytest.mark.parametrize('stamp', [None, 'other'])
def test_resume_refuses_checkpoints_from_other_inputs(tmp_path, stamp):
    ckpt = tmp_path / 'checkpoints'
    (ckpt / 'checkpoint-100').mkdir(parents=True)
    if stamp:
        (ckpt / 'inputs.sha256').write_text(stamp)
    with pytest.raises(ValueError, match='other training data or settings'):
        train.resume_checkpoint(ckpt, 'abc')


def test_fingerprint_tracks_data_and_training_settings_but_not_save_cadence(tmp_path):
    cfg = Config.model_validate(_cfg_dict(tmp_path))
    base = train.inputs_fingerprint(cfg, [example(3)])
    assert train.inputs_fingerprint(cfg, [example(4)]) != base
    assert train.inputs_fingerprint(cfg.model_copy(update={'train': cfg.train.model_copy(update={'learning_rate': 3e-4})}),
                                    [example(3)]) != base
    assert train.inputs_fingerprint(cfg.model_copy(update={'train': cfg.train.model_copy(update={'save_steps': 50})}),
                                    [example(3)]) == base


@pytest.mark.parametrize('save_steps', [None, 100])
def test_run_wires_checkpointing_into_the_trainer(tmp_path, monkeypatch, save_steps):
    transformers = pytest.importorskip('transformers')
    trl = pytest.importorskip('trl')
    pytest.importorskip('torch')
    d = _cfg_dict(tmp_path); d['train'].update(load_in_4bit=False, bf16=False, fp16=False, save_steps=save_steps)
    cfg = Config.model_validate(d); cfg.run_dir.mkdir()
    write_jsonl(cfg.run_dir / 'verified.jsonl', [{'question': 'Synthetic Q', 'answer': 'Synthetic A'}])

    class Tokenizer(WordTokenizer):
        def save_pretrained(self, *a): pass
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    seen = {}

    class Trainer:
        def __init__(self, **kw): seen['args'] = kw['args']
        def train(self, resume_from_checkpoint=None): seen['resume'] = resume_from_checkpoint
        def save_model(self, p): pass
    monkeypatch.setattr(trl, 'SFTTrainer', Trainer)

    if save_steps:  # a killed earlier attempt left a checkpoint for the same inputs
        train.run(cfg)
        (cfg.run_dir / 'checkpoints' / 'checkpoint-200').mkdir()
    train.run(cfg)
    assert seen['args'].save_strategy == ('no' if save_steps is None else 'steps')
    if save_steps:
        assert seen['args'].save_steps == 100
        assert seen['resume'] == str(cfg.run_dir / 'checkpoints' / 'checkpoint-200')
    else:
        assert seen['resume'] is None and not (cfg.run_dir / 'checkpoints' / 'inputs.sha256').exists()
