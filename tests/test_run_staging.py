"""Tests for the staged run body (L-01) and the frozen paper config.

Two things are covered here:

* **L-01 staging.** ``_run_body`` was split into ``_select_and_classify_rep``,
  ``_persist_rep`` and ``_finalize_run``. The point of the split is that a
  stage can be exercised *without* running a 25-repeat experiment, so these
  tests drive the stages directly with synthetic data.

* **Paper config.** The published numbers are only the paper's numbers if the
  config that produced them is the config that is shipped. ``PaperConfig``
  pins the published values and ``paper_drift`` reports, field by field, what
  moved. A drift report that does not say *what* moved is useless, so the
  tests assert on the message, not just on the fact of failure.

Offline: no network, no model, no qiskit, no training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quobo import run_experiment as RE
from src.quobo.config_schema import (
    PAPER_ARMS,
    PAPER_CLASSES,
    PaperConfig,
    load_paper_config,
    paper_drift,
    validate_paper_config,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _cfg(**overrides) -> dict:
    """A minimal config shaped like the paper's, for the staged functions."""
    cfg = {
        "seed": 42,
        "data": {"classes": list(PAPER_CLASSES), "min_images_per_class": 40},
        "features": {"backbone": "MobileNetV2", "pca_components": 50},
        "selection": {"k": 8},
        "qubo": {"mi_bins": 16, "sa_sweeps": 20000, "sa_repeats": 10, "parallel": False},
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 400,
            "test_fraction": 0.25,
            "tune_rbf": True,
            "max_qsvm_features": 20,
        },
        "experiment": {
            "n_repeats": 2,
            "arms": ["A_random", "D_qubo"],
            "early_stop": False,
            "track_mlflow": False,
        },
    }
    cfg.update(overrides)
    return cfg


def _ctx(n_train: int = 30, n_test: int = 10, n_features: int = 12):
    """A synthetic repeat context with plausible shapes."""
    rng = np.random.default_rng(7)
    Xtr = rng.normal(size=(n_train, n_features)).astype(np.float32)
    Xte = rng.normal(size=(n_test, n_features)).astype(np.float32)
    ytr = np.array(["a", "b"] * (n_train // 2))
    yte = np.array(["a", "b"] * (n_test // 2))
    return RE._RepeatContext(
        rep=0,
        seed=42,
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        classes=["a", "b"],
        y_test=[str(v) for v in yte],
        test_ids=[f"/crops/{c}/photo{i}_ann{i:04d}.jpg" for i, c in enumerate(yte)],
        test_groups=[f"photo{i}" for i in range(len(yte))],
    )


@pytest.fixture()
def stub_classifiers(monkeypatch):
    """Replace the real classifier stack with a deterministic stub.

    The staged functions must be testable without training an SVM and a QSVC.
    """

    def fake_all_classifiers(Xtr, ytr, Xte, yte, cfg, rep_offset=0):
        return {
            "rbf_svm": {
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "fit_seconds": 0.01,
                "predict_seconds": 0.01,
                "predictions": [str(v) for v in yte],
            }
        }

    monkeypatch.setattr(RE, "run_all_classifiers", fake_all_classifiers)


@pytest.fixture()
def stub_select(monkeypatch):
    """Select the first k columns for every arm (no real selection work)."""

    def fake_select(arm, X, y, k, seed, qubo_cfg):
        return list(range(k)), {"stub": True}

    monkeypatch.setattr(RE, "select_features", fake_select)


# --------------------------------------------------------------------------- #
# L-01: the staged run body
# --------------------------------------------------------------------------- #


def test_select_and_classify_rep_returns_rows_and_selection(stub_classifiers, stub_select):
    ctx = _ctx()
    cfg = _cfg()
    result = RE._select_and_classify_rep(ctx, "D_qubo", cfg)

    assert isinstance(result, RE._ArmResult)
    assert result.selected == list(range(cfg["selection"]["k"]))
    assert result.selection_info == {"stub": True}
    assert "rbf_svm" in result.metrics
    # one row per classifier, carrying the provenance columns the aggregates need
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["repeat"] == ctx.rep
    assert row["seed"] == ctx.seed
    assert row["arm_key"] == "D_qubo"
    assert row["classifier"] == "rbf_svm"
    assert row["selected"] == ",".join(map(str, result.selected))
    # predictions must not leak into the aggregate rows (they would bloat the CSV)
    assert "predictions" not in row


def test_z_full_arm_selects_every_component(stub_classifiers, stub_select):
    """The Z_full ceiling takes no selection path at all."""
    ctx = _ctx(n_features=12)
    cfg = _cfg()
    result = RE._select_and_classify_rep(ctx, "Z_full", cfg)
    assert result.selected == list(range(ctx.Xtr.shape[1]))
    assert result.selection_info == {"no_selection": True}
    assert result.selection_seconds == 0.0


def test_persist_rep_writes_aligned_provenance(tmp_path, stub_classifiers, stub_select):
    """The artifact must record the held-out identity aligned with predictions."""
    ctx = _ctx()
    cfg = _cfg()
    result = RE._select_and_classify_rep(ctx, "D_qubo", cfg)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    RE._persist_rep(run_dir, "stamp", "D_qubo", result, ctx)

    payload = json.loads((run_dir / "rep0_D_qubo.json").read_text(encoding="utf-8"))
    n = len(ctx.test_ids)
    assert payload["test_ids"] == ctx.test_ids
    assert payload["test_groups"] == ctx.test_groups
    assert len(payload["y_test"]) == n
    for clf, m in payload["metrics"].items():
        assert len(m["predictions"]) == n, f"{clf} predictions misaligned"
    assert payload["arm"] == "D_qubo"
    assert payload["rep"] == 0


def test_persist_rep_writes_atomically(tmp_path, stub_classifiers, stub_select):
    """A half-written artifact must never be visible."""
    ctx = _ctx()
    result = RE._select_and_classify_rep(ctx, "D_qubo", _cfg())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    RE._persist_rep(run_dir, "stamp", "D_qubo", result, ctx)
    # no temp files left behind
    leftovers = [p.name for p in run_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_finalize_run_writes_aggregates(tmp_path, stub_classifiers, stub_select):
    """The finalizer writes the summary CSVs the dashboard and paper read."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = _ctx()
    rows: list[dict] = []
    for arm in ("A_random", "D_qubo"):
        res = RE._select_and_classify_rep(ctx, arm, _cfg())
        rows.extend(res.rows)
        RE._persist_rep(run_dir, "stamp", arm, res, ctx)

    cfg = _cfg()
    results = RE._finalize_run(cfg, run_dir, "stamp", rows)

    assert isinstance(results, pd.DataFrame)
    assert not results.empty


# --------------------------------------------------------------------------- #
# Paper config: the published values are enforced, not merely described
# --------------------------------------------------------------------------- #


def test_paper_config_accepts_the_shipped_paper_config():
    cfg = load_paper_config("configs/experiment_6class.yaml")
    assert isinstance(cfg, PaperConfig)
    assert tuple(cfg.data.classes) == PAPER_CLASSES
    assert tuple(cfg.experiment.arms) == PAPER_ARMS
    assert cfg.selection.k == 8
    assert cfg.features.pca_components == 50
    assert cfg.experiment.n_repeats == 25
    assert cfg.seed == 42


def test_paper_drift_is_empty_for_the_paper_config():
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
    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg[section][field] = bad
    drifts = paper_drift(cfg)
    assert drifts, f"drift in {section}.{field} was not reported"
    joined = " | ".join(drifts)
    assert field in joined, f"drift report does not name {field}: {joined}"


def test_paper_drift_detects_reordered_classes():
    """The published tables index classes by position, so order is part of it."""
    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["data"]["classes"] = list(reversed(PAPER_CLASSES))
    drifts = paper_drift(cfg)
    assert drifts
    assert any("classes" in d for d in drifts)


def test_paper_drift_detects_reordered_arms():
    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["experiment"]["arms"] = list(reversed(PAPER_ARMS))
    drifts = paper_drift(cfg)
    assert drifts
    assert any("arms" in d for d in drifts)


def test_validate_paper_config_raises_with_a_readable_message():
    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["selection"]["k"] = 12
    with pytest.raises(ValueError) as exc:
        validate_paper_config(cfg)
    msg = str(exc.value)
    assert "paper config drift" in msg
    assert "k" in msg


def test_paper_config_allows_the_deliberately_free_fields():
    """``parallel`` and ``track_mlflow`` change the run's cost, not its meaning."""
    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["qubo"]["parallel"] = True
    cfg["experiment"]["track_mlflow"] = True
    assert paper_drift(cfg) == []


def test_paper_config_rejects_unknown_keys():
    """The frozen config must not silently accept a new knob."""
    cfg = load_paper_config("configs/experiment_6class.yaml").model_dump()
    cfg["selection"]["secret_sauce"] = 1
    drifts = paper_drift(cfg)
    assert drifts
    assert any("secret_sauce" in d for d in drifts)
