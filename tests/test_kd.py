import gzip
import json
import math

import numpy as np
import pytest

from distillkit.config import TrainCfg
from distillkit.kd import assistant_segments, kd_targets, load_teacher_topk

torch = pytest.importorskip("torch")
from distillkit.kd import base_kl_loss, sparse_kd_loss  # noqa: E402

STOP = {9}  # stands in for <|im_end|>


def write_topk(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def row(rid, toks, k=2):
    t = [f"token_id:{x}" for x in toks]
    top = [[[f"token_id:{x}", -0.1], [f"token_id:{x + 100}", -2.5]][:k] for x in toks]
    return {"id": rid, "tokens": t, "logprobs": [-0.1] * len(toks), "top": top}


def test_load_drops_stop_tokens_and_keys_by_answer_tokens(tmp_path):
    p = tmp_path / "lp.jsonl.gz"
    write_topk(p, [row("a", [5, 6, 9]), row("b", [7])])
    table = load_teacher_topk(p, STOP)
    assert set(table) == {(5, 6), (7,)}
    ids, lps = table[(5, 6)]
    assert ids.tolist() == [[5, 105], [6, 106]]
    assert lps.dtype == np.float32 and lps.shape == (2, 2)


def test_assistant_segments_finds_each_run():
    labels = torch.tensor([[-100, 5, 6, -100, 7, 9], [-100, -100, -100, -100, -100, -100]])
    assert assistant_segments(labels) == [(0, 1, 3), (0, 4, 6)]


def test_kd_targets_shift_by_one_and_skip_unmatched():
    table = {(5, 6): (np.array([[5, 105], [6, 106]]), np.zeros((2, 2), np.float32))}
    labels = torch.tensor([[-100, -100, 5, 6, 9, -100], [-100, 8, 8, 9, -100, -100]])
    rows, cols, ids, _ = kd_targets(labels, table, STOP)
    assert rows == [0, 0] and cols == [1, 2]  # logits at 1 and 2 predict the tokens at 2 and 3
    assert ids.tolist() == [[5, 105], [6, 106]]


def test_kd_targets_none_when_nothing_matches():
    labels = torch.tensor([[-100, 8, 8, 9]])
    assert kd_targets(labels, {}, STOP) is None


def test_truncated_segment_gets_no_kd():
    table = {(5, 6, 7): (np.zeros((3, 2), np.int64), np.zeros((3, 2), np.float32))}
    labels = torch.tensor([[-100, 5, 6]])  # cut at max_length before 7
    assert kd_targets(labels, table, STOP) is None


def test_sparse_kd_zero_when_student_matches_teacher():
    lp = torch.log(torch.tensor([[0.7, 0.3]]))
    logits = torch.full((1, 4), -1e4)
    logits[0, 1], logits[0, 3] = lp[0, 0], lp[0, 1]
    assert sparse_kd_loss(logits, torch.tensor([[1, 3]]), lp).item() == pytest.approx(0.0, abs=1e-6)


def test_sparse_kd_known_value():
    p = [0.75, 0.25]
    lp = torch.log(torch.tensor([p]))
    logits = torch.zeros((1, 3))  # student uniform over the 2 top ids
    want = sum(x * math.log(x / 0.5) for x in p)
    assert sparse_kd_loss(logits, torch.tensor([[0, 2]]), lp).item() == pytest.approx(want, rel=1e-5)


def test_sparse_kd_temperature_scales_by_t_squared():
    lp = torch.log(torch.tensor([[0.9, 0.1]]))
    logits = torch.zeros((1, 2))
    t = 2.0
    p = torch.softmax(lp / t, -1)[0]
    want = float((p * torch.log(p / 0.5)).sum()) * t * t
    assert sparse_kd_loss(logits, torch.tensor([[0, 1]]), lp, t).item() == pytest.approx(want, rel=1e-5)


def test_base_kl_zero_for_same_logits_and_positive_otherwise():
    a = torch.randn(3, 5)
    assert base_kl_loss(a, a).item() == pytest.approx(0.0, abs=1e-6)
    assert base_kl_loss(a, torch.randn(3, 5)).item() > 0


def test_config_defaults_are_plain_sft():
    c = TrainCfg()
    assert c.kd_alpha == 0 and c.base_kl == 0 and c.kd_file is None


def test_config_kd_needs_file():
    with pytest.raises(ValueError, match="kd_file"):
        TrainCfg(kd_alpha=0.5)
    assert TrainCfg(kd_alpha=0.5, kd_file="x.jsonl.gz").kd_alpha == 0.5
