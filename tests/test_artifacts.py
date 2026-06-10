import dataclasses
import json

import pandas as pd

from bmx.artifacts import create_run, write_metrics


@dataclasses.dataclass
class _Cfg:
    layers: tuple[int, ...] = (0, 1)
    device: str = "cpu"


def test_create_run_writes_config_and_env(tmp_path):
    run = create_run("unit_test_exp", _Cfg(), root=tmp_path)
    assert run.is_dir()
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["device"] == "cpu" and cfg["layers"] == [0, 1]
    env = json.loads((run / "env.json").read_text())
    assert "torch" in env and "git_sha" in env


def test_write_metrics_roundtrip(tmp_path):
    run = create_run("unit_test_exp", _Cfg(), root=tmp_path)
    df = pd.DataFrame([{"layer": 0, "method": "bmd_rals", "rel_error": 0.5}])
    write_metrics(run, df)
    back = pd.read_parquet(run / "metrics.parquet")
    assert back.iloc[0]["method"] == "bmd_rals"
