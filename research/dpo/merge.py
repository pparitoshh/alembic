"""Merge a (scaled) LoRA adapter into Qwen3-4B so DPO can train a fresh LoRA on top of it.

usage: merge.py ADAPTER_DIR OUT_DIR
"""
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter, out = sys.argv[1], sys.argv[2]
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Instruct-2507', dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print('merged', adapter, '->', out)
