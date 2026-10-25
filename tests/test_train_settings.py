"""Opt-in training settings from the mentor's recommendations (issue #37): validation interval and size, Liger."""
import importlib

import distillkit.train as train


def test_defaults_keep_the_previous_behaviour(monkeypatch):
    for k in ("EVAL_STEPS", "VAL_MAX_EXAMPLES", "USE_LIGER_KERNEL"):
        monkeypatch.delenv(k, raising=False)
    t = importlib.reload(train)
    assert t.EVAL_STEPS is None and t.VAL_MAX_EXAMPLES is None and t.USE_LIGER is False


def test_environment_opt_ins(monkeypatch):
    monkeypatch.setenv("EVAL_STEPS", "500")
    monkeypatch.setenv("VAL_MAX_EXAMPLES", "300")
    monkeypatch.setenv("USE_LIGER_KERNEL", "1")
    t = importlib.reload(train)
    assert (t.EVAL_STEPS, t.VAL_MAX_EXAMPLES, t.USE_LIGER) == (500, 300, True)
    monkeypatch.delenv("EVAL_STEPS")
    monkeypatch.delenv("VAL_MAX_EXAMPLES")
    monkeypatch.delenv("USE_LIGER_KERNEL")
    importlib.reload(train)


def test_train_passes_them_to_the_trainer():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src/distillkit/train.py").read_text()
    assert "eval_steps=EVAL_STEPS or tcfg.save_steps or 100" in src
    assert "use_liger_kernel=USE_LIGER" in src
    assert "val_ex = val_ex[:VAL_MAX_EXAMPLES]" in src
