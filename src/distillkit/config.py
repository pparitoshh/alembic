from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(path: str | Path) -> dict:
    load_dotenv()  # picks up API keys from ./.env; real env vars take precedence
    with open(path) as f:
        cfg = yaml.safe_load(f)
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
