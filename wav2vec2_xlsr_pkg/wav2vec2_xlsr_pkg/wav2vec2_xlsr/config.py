from __future__ import annotations
import argparse, yaml
from typing import Any, Dict
from pathlib import Path

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--config", type=str, default="configs/defaults.yaml")
    p.add_argument("--override", type=str, nargs="*", default=[],
                   help="Overrides like train.lr=1e-4 data.sample_rate=16000")
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    return p

def merge_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    for item in overrides:
        key, _, val = item.partition("=")
        keys = key.split(".")
        cursor = cfg
        for k in keys[:-1]:
            cursor = cursor.setdefault(k, {})
        # basic type casting
        if val.lower() in {"true","false"}:
            v = val.lower() == "true"
        else:
            try:
                v = int(val)
            except ValueError:
                try:
                    v = float(val)
                except ValueError:
                    v = val
        cursor[keys[-1]] = v
    return cfg
