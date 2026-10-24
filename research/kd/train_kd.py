"""Logit distillation, step 2: LoRA student trained on the teacher's top-20 distributions (GOAL.md G13).

usage: train_kd.py CONFIG.json KD_DIR [ALPHA]    (torchrun, one node = one seed, 4-GPU DDP)
Same rows, LoRA (r16/a32, all-linear), learning rate, schedule, epochs and global batch 16 as the SFT rounds;
only the loss differs: (1 - ALPHA) * cross-entropy on the answer + ALPHA * soft cross-entropy against the
teacher's renormalised top-20 next-token distribution at every answer token (ALPHA 0.5). Per-device batch 1 with
gradient accumulation; logits are computed for the answer positions only (they end each sequence).
Writes <run_dir>/adapter and TRAINING_DONE.json; resumes from the newest checkpoint.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from distillkit.config import load_config

cfg, kd_dir = load_config(sys.argv[1]), Path(sys.argv[2])
alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
tcfg, run_dir = cfg.train, cfg.run_dir
if (run_dir / 'adapter' / 'adapter_config.json').exists():
    print('[kd] adapter already trained:', run_dir)
    sys.exit(0)
tok = AutoTokenizer.from_pretrained(cfg.student.model)

teacher = {}  # row id -> (top-20 ids, log-probs) per answer token; ALPHA 0 = plain fine-tuning, no teacher scores
for f in sorted(kd_dir.glob('kd_*.npz')) if alpha > 0 else []:
    z = np.load(f)
    # Read each array once: z['ids'] re-reads the whole array on every access (per row, that OOM-killed the job).
    ids, lps, offsets, row_ids = z['ids'], z['lps'], z['offsets'], z['row_ids']
    for i, rid in enumerate(row_ids):
        teacher[str(rid)] = (ids[offsets[i]:offsets[i + 1]], lps[offsets[i]:offsets[i + 1]])


MIXED = os.environ.get('KD_MIXED') == '1'


class KDData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.rows = []
        for line in open(path):
            row = json.loads(line)
            # KD_MIXED=1 (opt-in, round 43): rows without teacher scores (e.g. tool workflows) are kept and trained
            # with the plain loss, while scored rows get the distillation loss.
            if alpha == 0 or row['id'] in teacher or MIXED:
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        # Same rendering as research/kd/score.py: prompt and answer tokenized separately, as at generation time.
        msgs = [{'role': 'system', 'content': cfg.student.system_prompt}, *row['messages']]
        # Tool rows (round 38) carry their tool schemas; rendered into the prompt as at inference. None = unchanged.
        prompt = tok.apply_chat_template(msgs[:-1], tools=row.get("tools"), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        prompt = tok(prompt, add_special_tokens=False)['input_ids']
        answer = tok(msgs[-1]['content'] + '<|im_end|>', add_special_tokens=False)['input_ids']
        full = prompt + answer
        n = len(answer)  # answer tokens, the tail of the sequence
        has_t = alpha > 0 and row['id'] in teacher
        if not has_t:
            ids, lps = np.zeros((n, 1), np.int64), np.zeros((n, 1), np.float32)
        else:
            ids, lps = teacher[row['id']]
            assert len(ids) == n, row['id']
        return {'input_ids': torch.tensor(full), 'n_answer': n, 'has_t': has_t,
                't_ids': torch.from_numpy(ids.astype(np.int64)), 't_lps': torch.from_numpy(lps.astype(np.float32))}


def collate(batch):
    assert len(batch) == 1, 'per-device batch 1'
    b = batch[0]
    return {'input_ids': b['input_ids'][None], 'n_answer': b['n_answer'], 'has_t': b['has_t'], 't_ids': b['t_ids'],
            't_lps': b['t_lps']}


class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ids, n = inputs['input_ids'], inputs['n_answer']
        # Logits at positions L-n-1 .. L-2 predict the n answer tokens.
        out = model(input_ids=ids, logits_to_keep=n + 1)
        logits = out.logits[0, :-1].float()  # (n, V)
        logp = F.log_softmax(logits, dim=-1)
        target = ids[0, -n:]
        ce = F.nll_loss(logp, target)
        if alpha == 0 or not inputs['has_t']:
            loss = ce
        else:
            t_prob = F.softmax(inputs['t_lps'], dim=-1)  # renormalised over the teacher's top-20
            soft = -(t_prob * logp.gather(1, inputs['t_ids'])).sum(-1).mean()
            loss = (1 - alpha) * ce + alpha * soft
        return (loss, out) if return_outputs else loss

    def _get_train_sampler(self, *args, **kwargs):
        # CURRICULUM=1 (opt-in, GOAL.md G14): keep the data file's order (sorted easy -> hard) instead of shuffling.
        import os
        if os.environ.get('CURRICULUM') == '1':
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)


ds = KDData(run_dir / 'verified.jsonl')
print(f'[kd] {len(ds)} rows with teacher distributions (alpha={alpha}), student={cfg.student.model}', flush=True)
model = AutoModelForCausalLM.from_pretrained(cfg.student.model, dtype=torch.bfloat16)
model.enable_input_require_grads()  # LoRA + gradient checkpointing: inputs must carry grad
model = get_peft_model(model, LoraConfig(r=tcfg.lora_r, lora_alpha=tcfg.lora_alpha, lora_dropout=tcfg.lora_dropout,
                                         target_modules='all-linear', task_type='CAUSAL_LM'))
world = max(1, torch.cuda.device_count())
# FSDP=1 (opt-in, GOAL.md G11): fully sharded data parallel over all GPUs of the job instead of DDP.
fsdp_kw = dict(fsdp='full_shard auto_wrap', fsdp_config={'transformer_layer_cls_to_wrap': ['Qwen3DecoderLayer'],
                                                          'use_orig_params': True}) if os.environ.get('FSDP') == '1' else {}
args = TrainingArguments(**fsdp_kw,
    output_dir=str(run_dir / 'checkpoints'), num_train_epochs=tcfg.epochs,
    per_device_train_batch_size=1, gradient_accumulation_steps=16 // world,  # global batch 16, as SFT
    learning_rate=tcfg.learning_rate, lr_scheduler_type='cosine', warmup_steps=0.03, bf16=True,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
    logging_steps=1, save_strategy='steps', save_steps=50, save_total_limit=2, seed=tcfg.seed,
    report_to='none', remove_unused_columns=False, dataloader_num_workers=2, ddp_find_unused_parameters=False)
trainer = KDTrainer(model=model, args=args, train_dataset=ds, data_collator=collate)
# Only complete checkpoints: a piece cut off by the time limit while saving leaves one without trainer_state.json.
ckpts = sorted((d for d in (run_dir / 'checkpoints').glob('checkpoint-*') if (d / 'trainer_state.json').exists()),
               key=lambda d: int(d.name.split('-')[-1]))
trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
if trainer.is_world_process_zero():
    trainer.model.save_pretrained(run_dir / 'adapter')
    tok.save_pretrained(run_dir / 'adapter')
    (run_dir / 'TRAINING_DONE.json').write_text(json.dumps({'config': sys.argv[1], 'kd_dir': str(kd_dir), 'alpha': alpha}))
