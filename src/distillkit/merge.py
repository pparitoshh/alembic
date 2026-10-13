"""Scaled merge of a (Q)DoRA/LoRA adapter: W = W_base + s * (W_merged - W_base), saved in bf16.

s = 1 is the plain merge; s = 0.2 keeps 20% of the fine-tuning change (WiSE-FT style, plan D5 start model).
Usage: python -m distillkit.merge BASE ADAPTER OUT --scale 0.2
"""

import argparse
from pathlib import Path


def scale_state(w0: dict, wm: dict, scale: float) -> dict:
    """Tensors of base + scale * (merged - base) for every tensor that the merge changed."""
    import torch

    out = {}
    for k, v in wm.items():
        if not torch.equal(v, w0[k]):
            out[k] = (w0[k].float() + scale * (v.float() - w0[k].float())).to(v.dtype)
    return out


def run(base: str, adapter: str, out: Path, scale: float) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (out / "config.json").exists():
        raise FileExistsError(f"{out} already holds a model; choose a new directory")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    w0 = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    if scale != 1.0:
        sd = model.state_dict()
        new = scale_state(w0, {k: v.detach().cpu() for k, v in sd.items()}, scale)
        with torch.no_grad():
            for k, v in new.items():
                sd[k].copy_(v)
        print(f"[merge] scaled {len(new)} tensors by {scale}")
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print(f"[merge] {adapter} x{scale} -> {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m distillkit.merge", description=__doc__.splitlines()[0])
    p.add_argument("base")
    p.add_argument("adapter")
    p.add_argument("out", type=Path)
    p.add_argument("--scale", type=float, default=1.0)
    a = p.parse_args()
    run(a.base, a.adapter, a.out, a.scale)


if __name__ == "__main__":
    main()
