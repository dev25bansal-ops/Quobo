from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from .config_schema import QuoboConfig

ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger(__name__)

# Keys the pipeline actually reads. Anything else in a config file is a typo
# or a leftover — fail loud rather than silently ignore it (OPEN-4).
KNOWN_KEYS = {
    "seed",
    "data",                      # classes, min_images_per_class (data_prep only)
    "features",                  # backbone, pca_components
    "selection",                 # k
    "qubo",                      # mi_bins, sa_sweeps, sa_repeats
    "qsvm",                      # reps, entanglement, max_train_samples,
                                 # test_fraction, tune_rbf, max_qsvm_features
    "experiment",                # n_repeats, arms
}


def load_config(path: str = "configs/experiment.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    unknown = set(cfg) - KNOWN_KEYS
    if unknown:
        log.warning("config %s has unrecognized top-level keys (ignored): %s", path, sorted(unknown))

    known_subkeys = {
        "features": {"backbone", "pca_components"},
        "qsvm": {"reps", "entanglement", "max_train_samples", "test_fraction",
                 "tune_rbf", "max_qsvm_features"},
        "experiment": {"n_repeats", "arms", "early_stop", "track_mlflow"},
        "selection": {"k"},
        "qubo": {"mi_bins", "sa_sweeps", "sa_repeats", "parallel"},
        "data": {"classes", "min_images_per_class", "crops_dir_override"},
    }
    for section, keys in known_subkeys.items():
        if section in cfg and isinstance(cfg[section], dict):
            dead = set(cfg[section]) - keys
            if dead:
                log.warning("config %s: %s has dead keys (never read by code): %s",
                            path, section, sorted(dead))
    return cfg


def validate_config(cfg: dict) -> QuoboConfig:
    """Enforce the full schema (config_schema.QuoboConfig). Raises ValueError
    with a readable message on malformed keys / out-of-range values / bad arm
    names. Call this at the top of any long run to fail fast, not at rep 12."""
    from .config_schema import QuoboConfig

    try:
        return QuoboConfig.model_validate(cfg)
    except Exception as e:  # pydantic.ValidationError
        raise ValueError(f"invalid experiment config: {e}") from e


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
