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


# ---------- PaperConfig: the published configuration is enforced ----------

# The paper's numbers are only the paper's numbers if the config that produced
# them is the config that is shipped. `PaperConfig` pins the published values
# and `paper_drift` reports, field by field, what moved. A drift report that
# does not say *what* moved is useless, so the tests assert on the message,
# not just on the fact of failure.


def test_paper_config_accepts_the_shipped_paper_config():
    from src.quobo.config_schema import PAPER_ARMS, PAPER_CLASSES, PaperConfig, load_paper_config

    cfg = load_paper_config("configs/experiment_6class.yaml")
    assert isinstance(cfg, PaperConfig)
    assert tuple(cfg.data.classes) == PAPER_CLASSES
    assert tuple(cfg.experiment.arms) == PAPER_ARMS
    assert cfg.seed == 42
    assert cfg.selection.k == 8
    assert cfg.features.pca_components == 50
    assert cfg.experiment.n_repeats == 25


def test_paper_drift_is_empty_for_the_paper_config():
    from src.quobo.config_schema import load_paper_config, paper_drift

    cfg = load_paper_config("configs/experiment_6class.yaml")
    assert paper_drift(cfg.model_dump()) == []


@pytest.mark.parametrize(
    "section, field, bad",
    [
        ("selection", "k", 10),
        ("features", "pca_components", 30),
        ("qubo", "mi_bins", 32),
        ("qsvm", "test_fraction", 0.5),
        ("experiment", "n_repeats", 10),
    ],
)
def test_paper_drift_names_the_drifted_field(section, field, bad):
    """A drift report must say WHICH field moved, not just that something did."""
    from src.quobo.config_schema import load_paper_config, paper_drift

    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg[section][field] = bad
    drifts = paper_drift(cfg)
    assert drifts, f"drift in {section}.{field} was not reported"
    joined = " | ".join(drifts)
    assert field in joined, f"drift report does not name {field}: {joined}"


def test_paper_drift_detects_reordered_classes():
    """The published tables index classes by position, so order is part of it."""
    from src.quobo.config_schema import PAPER_CLASSES, load_paper_config, paper_drift

    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["data"]["classes"] = list(reversed(PAPER_CLASSES))
    drifts = paper_drift(cfg)
    assert drifts
    assert any("classes" in d for d in drifts)


def test_paper_drift_detects_reordered_arms():
    from src.quobo.config_schema import PAPER_ARMS, load_paper_config, paper_drift

    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["experiment"]["arms"] = list(reversed(PAPER_ARMS))
    drifts = paper_drift(cfg)
    assert drifts
    assert any("arms" in d for d in drifts)


def test_validate_paper_config_raises_with_a_readable_message():
    from src.quobo.config_schema import load_paper_config, validate_paper_config

    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["selection"]["k"] = 12
    with pytest.raises(ValueError) as exc:
        validate_paper_config(cfg)
    msg = str(exc.value)
    assert "paper config drift" in msg
    assert "k" in msg


def test_paper_config_allows_the_deliberately_free_fields():
    """``parallel`` and ``track_mlflow`` change the run's cost, not its meaning."""
    from src.quobo.config_schema import load_paper_config, paper_drift

    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["qubo"]["parallel"] = True
    cfg["experiment"]["track_mlflow"] = True
    assert paper_drift(cfg) == []


def test_paper_config_rejects_unknown_keys():
    """The frozen config must not silently accept a new knob."""
    from src.quobo.config_schema import load_paper_config, paper_drift

    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["selection"]["secret_sauce"] = 1
    drifts = paper_drift(cfg)
    assert drifts
    assert any("secret_sauce" in d for d in drifts)
