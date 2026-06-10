"""Run-directory management: every experiment writes config + env + parquet
metrics under results/<experiment>/<run-id>/ (framework extension point #3)."""

import dataclasses
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_run(experiment: str, config, root="results") -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{git_sha()}"
    run = Path(root) / experiment / run_id
    run.mkdir(parents=True, exist_ok=False)

    if dataclasses.is_dataclass(config):
        cfg = dataclasses.asdict(config)
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = vars(config)
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    env = {
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("tensorly", "transformers", "numpy", "pandas")
        },
    }
    (run / "env.json").write_text(json.dumps(env, indent=2))
    return run


def write_metrics(run_dir: Path, df: pd.DataFrame, name: str = "metrics") -> Path:
    out = Path(run_dir) / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out
