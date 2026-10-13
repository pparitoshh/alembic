"""Extra SFT loss terms: sparse logit KD from the teacher top-k, and a KL penalty towards the base model.

Teacher top-k rows (`teacher_logprobs.jsonl.gz`, written by `generate`) are matched to the training batch by
the answer token ids: student and teacher share the Qwen3 tokenizer, and the training subset keeps only rows
whose teacher tokens equal the student tokenization of the answer. A segment without a match (no teacher row,
or truncated at max_length) gets CE only.
"""

import gzip
import json
from pathlib import Path

import numpy as np


def _token_id(t: str) -> int:
    return int(t.split(":", 1)[1])  # vLLM return_tokens_as_token_ids: "token_id:<n>"


def load_teacher_topk(path: Path, stop_ids: set[int]) -> dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]]:
    """answer token ids (stop tokens removed) -> (top ids [n, k] int64, top logprobs [n, k] float32)."""
    table = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            keep = [i for i, t in enumerate(d["tokens"]) if _token_id(t) not in stop_ids]
            key = tuple(_token_id(d["tokens"][i]) for i in keep)
            top = [d["top"][i] for i in keep]
            k = min(len(t) for t in top) if top else 0
            if not k or key in table:
                continue
            ids = np.array([[_token_id(tok) for tok, _ in t[:k]] for t in top], dtype=np.int64)
            lps = np.array([[lp for _, lp in t[:k]] for t in top], dtype=np.float32)
            table[key] = (ids, lps)
    return table


def assistant_segments(labels) -> list[tuple[int, int, int]]:
    """(row, start, end) of each contiguous run of label positions (labels != -100) in a [B, L] tensor."""
    out = []
    for b, row in enumerate(labels.tolist()):
        start = None
        for p, v in enumerate(row + [-100]):
            if v != -100 and start is None:
                start = p
            elif v == -100 and start is not None:
                out.append((b, start, p))
                start = None
    return out


def kd_targets(labels, table, stop_ids: set[int]):
    """Positions of the logits that predict matched teacher tokens, with their teacher top-k.

    Returns (rows, cols, top ids [N, k], top logprobs [N, k]) or None. The logit at position p-1 predicts
    the token at p. A segment ends at its first stop token (`<|im_end|>`); the teacher row has no stop tokens.
    """
    rows, cols, ids, lps = [], [], [], []
    lab = labels.tolist()
    for b, s, e in assistant_segments(labels):
        seg = lab[b][s:e]
        # TRL's assistant mask covers `<|im_end|>` and the newline after it: cut at the first stop token
        cut = next((i for i, t in enumerate(seg) if t in stop_ids), len(seg))
        seg = seg[:cut]
        hit = table.get(tuple(seg))
        if hit is None or s == 0:
            continue
        n = len(seg)
        rows += [b] * n
        cols += range(s - 1, s - 1 + n)
        ids.append(hit[0])
        lps.append(hit[1])
    if not rows:
        return None
    k = min(a.shape[1] for a in ids)
    return rows, cols, np.concatenate([a[:, :k] for a in ids]), np.concatenate([a[:, :k] for a in lps])


def sparse_kd_loss(student_logits, top_ids, top_logprobs, temperature: float = 1.0):
    """Mean KL(teacher ‖ student) over N positions, both renormalized over the teacher top-k ids.

    student_logits [N, V], top_ids [N, k] (long), top_logprobs [N, k]. Scaled by T² (Hinton et al.).
    """
    import torch

    t = temperature
    p_t = torch.softmax(top_logprobs.float() / t, dim=-1)
    s = torch.gather(student_logits.float() / t, 1, top_ids)
    log_q = torch.log_softmax(s, dim=-1)
    kl = (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_q)).sum(-1)
    return kl.mean() * t * t


def base_kl_loss(student_logits, base_logits):
    """Mean KL(student ‖ base) over N positions, full vocabulary. Inputs [N, V]."""
    import torch

    log_p = torch.log_softmax(student_logits.float(), dim=-1)
    log_b = torch.log_softmax(base_logits.float(), dim=-1)
    return (log_p.exp() * (log_p - log_b)).sum(-1).mean()
