import argparse
import os

from .config import load_config

# torch 2.14 routes some ops (e.g. RoPE's bmm) to Triton kernels that need a C compiler
# at first use; fall back to the stock kernels. Must be set before torch is imported.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

STAGES = ["generate", "verify", "train", "evaluate"]


def main() -> None:
    p = argparse.ArgumentParser(prog="distillkit", description="Teacher -> student sequence-level distillation")
    p.add_argument("stage", choices=STAGES + ["all"])
    p.add_argument("-c", "--config", default="configs/toy.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    for stage in STAGES if args.stage == "all" else [args.stage]:
        if stage == "generate":
            from . import generate as mod
        elif stage == "verify":
            from . import verify as mod
        elif stage == "train":
            from . import train as mod
        else:
            from . import evaluate as mod
        mod.run(cfg)


if __name__ == "__main__":
    main()
