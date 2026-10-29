"""Full-parameter distillation of the 4B student from teacher answers with top-20 log-probabilities (8 GPUs; data
parallel with an explicit gradient all-reduce after each step's sequences).

usage: torchrun --nproc_per_node 8 train_kd.py STUDENT_DIR DATA_DIR OUT_DIR
Loss on answer tokens only: ALPHA * cross-entropy against the teacher's renormalised top-20 distribution +
(1 - ALPHA) * cross-entropy on the teacher's sampled token. fp32 master weights, bf16 autocast, AdamW, cosine schedule.
Env: ALPHA (0.8), LR (5e-6), EPOCHS (1), ACCUM (8 sequences per GPU per step), MAX_LEN (12288), SEED (0).
"""
import glob
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

student, data, out = sys.argv[1:4]
ALPHA, LR = float(os.environ.get('ALPHA', 0.8)), float(os.environ.get('LR', 5e-6))
EPOCHS, ACCUM = int(os.environ.get('EPOCHS', 1)), int(os.environ.get('ACCUM', 8))
MAX_LEN, SEED = int(os.environ.get('MAX_LEN', 12288)), int(os.environ.get('SEED', 0))
dist.init_process_group('nccl')
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dev = torch.device('cuda')

# --- data: every (prompt, answer, top-k) triple from the teacher parts
items = []
for f in sorted(glob.glob(f'{data}/part*.npz')):
    z = np.load(f)
    po, oo = z['prompt_off'], z['out_off']
    for i in range(len(po) - 1):
        items.append((f, int(po[i]), int(po[i + 1]), int(oo[i]), int(oo[i + 1])))
cache = {}


def load(f):
    if f not in cache:
        z = np.load(f)
        cache[f] = {k: z[k] for k in ('prompt', 'out', 'topk_ids', 'topk_lp')}
    return cache[f]


rng = random.Random(SEED)
order = list(range(len(items)))
steps_per_epoch = len(order) // (world * ACCUM)
total = steps_per_epoch * EPOCHS
if rank == 0:
    print(f'[kd] {len(items)} sequences, {world} GPUs x {ACCUM} accum, {total} optimizer steps, alpha {ALPHA}, lr {LR}', flush=True)

model = AutoModelForCausalLM.from_pretrained(student, dtype=torch.float32, attn_implementation='sdpa').to(dev)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.config.use_cache = False
for prm in model.parameters():  # same start on every rank
    dist.broadcast(prm.data, 0)
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
warm = max(1, int(0.03 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, total)))))


def loss_of(it):
    f, p0, p1, o0, o1 = it
    d = load(f)
    p, o = d['prompt'][p0:p1], d['out'][o0:o1]
    ti, tl = d['topk_ids'][o0:o1], d['topk_lp'][o0:o1].astype(np.float32)
    keep = max(0, len(p) + len(o) - MAX_LEN)  # drop the start of over-long prompts (rare)
    ids = torch.from_numpy(np.concatenate([p[keep:], o]).astype(np.int64)).to(dev)[None]
    n_p = len(p) - keep
    with torch.autocast('cuda', dtype=torch.bfloat16):
        h = model.model(input_ids=ids).last_hidden_state[0, n_p - 1: n_p - 1 + len(o)]
        logits = model.lm_head(h).float()
    logp = Fn.log_softmax(logits, -1)
    tgt = torch.from_numpy(o.astype(np.int64)).to(dev)
    ce = -logp.gather(1, tgt[:, None]).squeeze(1)
    pt = torch.softmax(torch.from_numpy(tl).to(dev), -1)  # teacher top-20, renormalised
    kd = -(pt * logp.gather(1, torch.from_numpy(ti.astype(np.int64)).to(dev))).sum(-1)
    return (ALPHA * kd + (1 - ALPHA) * ce).sum(), len(o)


step, t0 = 0, time.time()
for ep in range(EPOCHS):
    rng.shuffle(order)
    for s in range(steps_per_epoch):
        batch = order[(s * world + rank) * ACCUM:(s * world + rank + 1) * ACCUM]
        ntok = torch.tensor([sum(items[i][4] - items[i][3] for i in batch)], device=dev, dtype=torch.float32)
        dist.all_reduce(ntok)
        tot = 0.0
        for i in batch:
            l, n = loss_of(items[i])
            (l / ntok).backward()  # summed over ranks below: the step is a mean over every answer token of the step
            tot += l.item()
        works = [dist.all_reduce(prm.grad, async_op=True) for prm in model.parameters() if prm.grad is not None]
        for w in works:
            w.wait()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if rank == 0 and (step % 10 == 0 or step == 1):
            print(f'[kd] step {step}/{total} loss/token {tot / max(1, ntok.item() / world):.4f} lr {sched.get_last_lr()[0]:.2e} '
                  f'{(time.time() - t0) / step:.1f}s/step', flush=True)
dist.barrier()
if rank == 0:
    os.makedirs(out, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True, max_shard_size='20GB')  # one model.safetensors, as on Hugging Face
    AutoTokenizer.from_pretrained(student).save_pretrained(out)
    for f in ('generation_config.json', 'chat_template.jinja'):
        if os.path.exists(f'{student}/{f}'):
            shutil.copy(f'{student}/{f}', out)
    json.dump({'student': student, 'data': data, 'alpha': ALPHA, 'lr': LR, 'epochs': EPOCHS, 'accum': ACCUM,
               'world': world, 'sequences': len(items), 'steps': total}, open(f'{out}/KD_TRAIN.json', 'w'), indent=1)
    print('[kd] saved', out, flush=True)
dist.destroy_process_group()
