import argparse
import importlib
import os

from .config import load_config

# torch 2.14 routes some ops (e.g. RoPE's bmm) to Triton kernels that need a C compiler
# at first use; fall back to the stock kernels. Must be set before torch is imported.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

# stage name -> module with `run(cfg)`, in pipeline order; modules import lazily so
# `generate` and `verify` work without the training extras installed.
# `answer` (student/base answers on GPU) and `evaluate` (checks + judge) are split so a cluster job can
# free the GPU before starting the judge server; `evaluate` runs `answer` itself if needed.
STAGES = {
    "generate": "generate",
    "verify": "verify",
    "train": "train",
    "answer": "evaluate",
    "evaluate": "evaluate",
    "export": "export",
}
EXTRA = {"calibrate": "calibrate"}  # not part of `all`


def main() -> None:
    p = argparse.ArgumentParser(prog="distillkit", description="Teacher -> student sequence-level distillation")
    p.add_argument("stage", choices=[*STAGES, *EXTRA, "all"])
    p.add_argument("-c", "--config", default="configs/toy_qdora.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set train.seed=1 --set run_dir=runs/x/seed1")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    for stage in STAGES if args.stage == "all" else [args.stage]:
        module = importlib.import_module(f".{(STAGES | EXTRA)[stage]}", __package__)
        module.run_answers(cfg) if stage == "answer" else module.run(cfg)


if __name__ == "__main__":
    main()
