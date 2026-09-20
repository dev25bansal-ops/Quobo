"""Unit tests for the config-schema validation gate (Enhancement 1)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _valid_cfg():
    return {
        "seed": 42,
        "data": {"classes": ["bottle", "can", "carton"], "min_images_per_class": 40},
        "features": {"backbone": "MobileNetV2", "pca_components": 50},
        "selection": {"k": 8},
        "qubo": {"mi_bins": 16, "sa_sweeps": 20000, "sa_repeats": 10},
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 400,
            "test_fraction": 0.25,
            "tune_rbf": True,
            "max_qsvm_features": 20,
        },
        "experiment": {
            "n_repeats": 25,
            "arms": ["A_random", "B_pca", "C_mi", "D_qubo", "E_lasso", "F_mrmr", "Z_full"],
        },
    }


def test_valid_config_passes():
    from src.quobo.config import validate_config

    schema = validate_config(_valid_cfg())
    assert schema.seed == 42
    assert schema.selection.k == 8


def test_invalid_arm_name_fails():
    from src.quobo.config import validate_config

    cfg = _valid_cfg()
    cfg["experiment"]["arms"] = ["A_random", "Z_typo"]
    with pytest.raises(ValueError, match="invalid experiment config"):
        validate_config(cfg)


def test_out_of_range_k_fails():
    from src.quobo.config import validate_config

    cfg = _valid_cfg()
    cfg["selection"]["k"] = 999  # > 50
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_negative_seed_fails():
    from src.quobo.config import validate_config

    cfg = _valid_cfg()
    cfg["seed"] = -1
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_duplicate_classes_fails():
    from src.quobo.config import validate_config

    cfg = _valid_cfg()
    cfg["data"]["classes"] = ["bottle", "bottle", "can"]
    with pytest.raises(ValueError):
        validate_config(cfg)
