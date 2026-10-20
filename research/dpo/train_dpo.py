"""DPO on top of the merged round-5 student: a new LoRA learns the selector's preference (chosen over rejected).

usage: train_dpo.py MERGED_MODEL_DIR TRAIN.jsonl OUT_DIR
The reference policy is the merged model itself (adapter disabled), so DPO only shifts preferences away from
the round-5 student. Resumes from the newest checkpoint (30-minute debug jobs). Writes OUT_DIR/adapter.
"""
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

model_dir, data, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if (out / 'adapter' / 'adapter_config.json').exists():
    sys.exit(0)
tok = AutoTokenizer.from_pretrained(model_dir)
ds = load_dataset('json', data_files=data, split='train')
args = DPOConfig(
    output_dir=str(out / 'checkpoints'), num_train_epochs=1, per_device_train_batch_size=1,
    gradient_accumulation_steps=2, learning_rate=5e-6, lr_scheduler_type='cosine', warmup_steps=0.05,
    beta=0.1, max_length=4096, bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=5, save_strategy='steps',
    save_steps=25, save_total_limit=2, report_to='none', seed=42,
    model_init_kwargs={'dtype': torch.bfloat16})
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules='all-linear', task_type='CAUSAL_LM')
trainer = DPOTrainer(model=model_dir, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
ckpts = sorted((out / 'checkpoints').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
if trainer.is_world_process_zero():
    trainer.model.save_pretrained(out / 'adapter')
    tok.save_pretrained(out / 'adapter')
    print('[dpo] adapter saved ->', out / 'adapter')
