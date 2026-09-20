from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

from .config_schema import QuoboConfig

ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger(__name__)

# Keys the pipeline actually reads. Anything else in a config file is a typo
# or a leftover — fail loud rather than silently ignore it (Q17).
#
# M-04: derived from the pydantic schema itself, so the allow-list can never
# drift from the model. Adding a field to config_schema automatically admits it
# here; the schema's extra="forbid" remains the authoritative gate.
KNOWN_KEYS = set(QuoboConfig.model_fields)

KNOWN_SUBKEYS = {
    name: set(field.annotation.model_fields)
    for name, field in QuoboConfig.model_fields.items()
    if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel)
}


def load_config(path: str = "configs/experiment_6class.yaml") -> dict:
    """Load a YAML config and reject unknown key *names*.

    Unknown top-level keys and unknown keys inside a known section raise
    ``ValueError`` (Q17) instead of being silently ignored — a mistyped key
    otherwise changes nothing and the run proceeds with defaults, hiding the
    error until late (or forever)."""
    with open(ROOT / path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"config {path} is empty")
    if not isinstance(cfg, dict):
        raise TypeError(f"config {path} must be a mapping at the top level")

    unknown = set(cfg) - KNOWN_KEYS
    if unknown:
        raise ValueError(
            f"config {path} has unknown top-level keys: {sorted(unknown)} "
            f"(known: {sorted(KNOWN_KEYS)})"
        )
    for section, keys in KNOWN_SUBKEYS.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            continue
        dead = set(cfg[section]) - keys
        if dead:
            raise ValueError(
                f"config {path}: section '{section}' has unknown keys: "
                f"{sorted(dead)} (known: {sorted(keys)})"
            )
    return cfg


def validate_config(cfg: dict) -> QuoboConfig:
    """Enforce the full schema (config_schema.QuoboConfig). Raises ValueError
    with a readable message on malformed keys / out-of-range values / bad arm
    names / unknown keys / cross-field violations. Call this at the top of any
    long run to fail fast, not at rep 12."""
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
