from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str = "configs/experiment.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> dict:
    dirs = {
        "raw": ROOT / "data" / "raw",
        "processed": ROOT / "data" / "processed",
        "features": ROOT / "data" / "features",
        "experiments": ROOT / "experiments",
        "figures": ROOT / "results" / "figures",
        "tables": ROOT / "results" / "tables",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
