"""One seed of the RAFT student on one node (DDP over its GPUs). Resumes from the newest checkpoint."""
import json
import sys
from pathlib import Path

from distillkit.config import load_config
from distillkit.train import run

cfg = load_config(sys.argv[1])
if (cfg.run_dir / 'adapter' / 'adapter_config.json').exists():
    print('[rag65] adapter already trained:', cfg.run_dir)
    sys.exit(0)
run(cfg)
import torch.distributed as dist
if not dist.is_initialized() or dist.get_rank() == 0:
    (cfg.run_dir / 'TRAINING_DONE.json').write_text(json.dumps({'config': sys.argv[1]}))
