import argparse
import importlib
import os

from .config import load_config

# torch 2.14 routes some ops (e.g. RoPE's bmm) to Triton kernels that need a C compiler
# at first use; fall back to the stock kernels. Must be set before torch is imported.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

# stage name -> module with `run(cfg)`, in pipeline order; modules import lazily so
# `generate` and `verify` work without the training extras installed
STAGES = {"generate": "generate", "verify": "verify", "train": "train", "evaluate": "evaluate", "export": "export"}


def main() -> None:
    p = argparse.ArgumentParser(prog="distillkit", description="Teacher -> student sequence-level distillation")
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("-c", "--config", default="configs/toy_qdora.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    for stage in STAGES if args.stage == "all" else [args.stage]:
        importlib.import_module(f".{STAGES[stage]}", __package__).run(cfg)


if __name__ == "__main__":
    main()
