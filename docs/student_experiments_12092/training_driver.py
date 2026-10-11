"""Train a separate student adapter on the certified 12,092-row release."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import argparse
import importlib.metadata
import os
import shutil
import sys
import unicodedata

BASE = Path('/leonardo_work/EUHPC_D30_031/runs/a08trc09/distillkit-20261006T231002Z')
HERE = BASE / 'dependency-pipeline-v1'
OUT = BASE / 'student-training-12092-v1'
TARGET = 12092
INTERIM = BASE / 'interim-training-1100-v1'
COHORTS = ('newdocs-seventh-answers-v2',)


def sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--output', default='student-training-12092-v1')
    args = ap.parse_args()
    assert args.seed in (42, 43, 44)
    assert args.learning_rate in (1e-4, 5e-5)
    assert '/' not in args.output and args.output.startswith('student-training-12092')
    OUT = BASE / args.output
    assert os.environ.get('SLURM_JOB_ID')
    release = None
    for cohort in COHORTS:
        cert = BASE / cohort / 'certification_v2'
        report_path = cert / 'report.json'
        if not report_path.exists():
            continue
        evidence = json.loads(report_path.read_text())
        candidate = cert / 'combined.jsonl'
        assert evidence['status'] == 'AUTOMATED_GATE_CERTIFIED'
        assert sha(candidate) == evidence['combined_sha256']
        if evidence['combined'] >= TARGET:
            release = candidate
            break
    if release is None:
        save(HERE / 'training_12092_wait.json', {'utc': utc(), 'status': 'WAITING_FOR_12092_VERIFIED'})
        print('Milestone training withheld: fewer than 12,092 automated-gate verified rows.')
        return

    OUT.mkdir(exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    dataset = OUT / 'verified.jsonl'
    if dataset.exists():
        assert sha(dataset) == sha(release)
    else:
        shutil.copyfile(release, dataset)
    # distillkit.train.run loads verified.jsonl from cfg.run_dir, not OUT.
    run_dataset = OUT / 'run/verified.jsonl'
    run_dataset.parent.mkdir(exist_ok=True)
    if run_dataset.exists():
        assert sha(run_dataset) == sha(dataset)
    else:
        shutil.copyfile(dataset, run_dataset)
    cfg_data = json.loads((INTERIM / 'config.json').read_text())
    cfg_data['run_dir'] = str(OUT / 'run')
    cfg_data['train']['seed'] = args.seed
    cfg_data['train']['learning_rate'] = args.learning_rate
    cfg_path = OUT / 'config.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == cfg_data
    else:
        save(cfg_path, cfg_data)
    sys.path.insert(0, str(INTERIM / 'code/src'))
    os.environ.update({'HF_HOME': '/leonardo_work/EUHPC_D30_031/hf_cache',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '8'})
    from distillkit.config import Config
    from distillkit.records import training_example
    from distillkit import train
    import torch
    from transformers import TrainerCallback, AutoTokenizer, set_seed
    set_seed(args.seed)
    model_ref = Path(os.environ['HF_HOME']) / 'hub/models--Qwen--Qwen3-4B-Instruct-2507/refs/main'
    assert model_ref.read_text().strip() == 'cdbee75f17c01a7cc42f958dc650907174af0554'

    rows = [json.loads(line) for line in dataset.open()]
    assert len(rows) >= TARGET
    assert len({r['id'] for r in rows}) == len(rows)
    norm = lambda q: ' '.join(unicodedata.normalize('NFKC', q).casefold().split())
    assert len({norm(r['question']) for r in rows}) == len(rows)
    assert all(r.get('teacher') == 'Qwen/Qwen3-32B-AWQ' for r in rows)
    cfg = Config.model_validate(cfg_data)
    assert cfg.train.method == 'dora' and cfg.train.load_in_4bit and cfg.train.bf16
    assert cfg.train.input_manifest is None and cfg.seeds.registry is None
    assert len([training_example(r, cfg.student.system_prompt) for r in rows]) == len(rows)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    result = {'status': 'RUNNING', 'started_utc': utc(), 'job_id': os.environ['SLURM_JOB_ID'],
              'source_release': str(release), 'source_release_sha256': sha(release),
              'records': len(rows), 'mode_counts': dict(Counter(r.get('mode') for r in rows)),
              'config_sha256': sha(cfg_path), 'loss_logs': [], 'optimizer_steps': 0,
              'seed_before_model_init': args.seed, 'training_config': cfg_data['train'],
              'model_revision': model_ref.read_text().strip(),
              'training_code_sha256': sha(Path(train.__file__)),
              'packages': {p: importlib.metadata.version(p) for p in ('torch','transformers','trl','peft','bitsandbytes')}}
    evidence_path = OUT / 'training_evidence.json'
    save(evidence_path, result)

    class Audit(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                assert torch.isfinite(torch.tensor(logs['loss']))
                result['loss_logs'].append({'step': state.global_step, **logs})
                save(evidence_path, result)

        def on_optimizer_step(self, args, state, control, **kwargs):
            result['optimizer_steps'] += 1
            if result['optimizer_steps'] % 10 == 0:
                save(evidence_path, result)

    def before(trainer):
        assert len(trainer.train_dataset) == len(rows)
        eos = trainer.processing_class.eos_token_id
        lengths, supervised = [], []
        for row, prepared in zip(rows, trainer.train_dataset, strict=True):
            ids, labels = prepared['input_ids'], prepared['labels']
            assert len(ids) == len(labels) and len(ids) < cfg.train.max_length, row['id']
            assert all(label == -100 or label == token for label, token in zip(labels, ids))
            selected = [x for x in labels if x != -100]
            assert selected and eos in selected and any(x == -100 for x in labels), row['id']
            lengths.append(len(ids)); supervised.append(len(selected))
        batch = trainer.data_collator([dict(trainer.train_dataset[i]) for i in range(32)])
        assert (batch['labels'][batch['attention_mask'] == 0] == -100).all()
        result['loader'] = {'rows': len(rows), 'min_tokens': min(lengths),
                            'max_tokens': max(lengths), 'total_tokens': sum(lengths),
                            'supervised_tokens': sum(supervised), 'eos_token_id': eos,
                            'all_rows_assistant_masked_and_eos_supervised': True,
                            'padding_masked': True, 'no_max_length_boundary_hit': True}
        save(evidence_path, result)

    adapter_path = OUT / 'run/adapter'
    if adapter_path.exists():
        raise RuntimeError('Milestone adapter already exists; refusing to overwrite it')
    try:
        adapter = train.run(cfg, before_train=before, callbacks=[Audit()])
        assert adapter.is_dir() and (adapter / 'adapter_model.safetensors').is_file()
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        assert tokenizer.eos_token_id == result['loader']['eos_token_id']
        result.update(status='PASS', adapter_path=str(adapter),
                      adapter_sha256=sha(adapter / 'adapter_model.safetensors'),
                      finished_utc=utc())
        save(evidence_path, result)
    except Exception as exc:
        result.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_utc=utc())
        save(evidence_path, result)
        raise


if __name__ == '__main__':
    main()
