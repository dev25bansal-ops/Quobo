"""Offline regression tests for the Q07/Q15/Q17/Q19/Q24 hardening items.

Everything here is network-free, cloud-free and model-free: TensorFlow/qiskit
are never loaded on the paths exercised, and all fakes are local.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quobo import features as F
from src.quobo import run_experiment as RE
from src.quobo.config import KNOWN_KEYS, KNOWN_SUBKEYS, load_config, validate_config
from src.quobo.selection import select_qubo

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _schema_cfg() -> dict:
    """Minimal schema-valid config (no optional keys) for validation tests."""
    return {
        "seed": 7,
        "data": {"classes": ["bottle", "cigarette"], "min_images_per_class": 10},
        "features": {"backbone": "MobileNetV2", "pca_components": 10},
        "selection": {"k": 2},
        "qubo": {"mi_bins": 4, "sa_sweeps": 1000, "sa_repeats": 5},
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 50,
            "test_fraction": 0.25,
            "tune_rbf": False,
        },
        "experiment": {"n_repeats": 5, "arms": ["D_qubo"]},
    }


def _raw_frame(n_groups: int = 6, per_group: int = 2, emb: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        label = "bottle" if g % 2 == 0 else "cigarette"
        for j in range(per_group):
            row = {f"e{i}": float(rng.normal()) for i in range(emb)}
            row["label"] = label
            row["crop_path"] = f"/crops/{label}/batch_{g}_0001_ann{j}.jpg"
            rows.append(row)
    return pd.DataFrame(rows)


def _fake_classifiers(Xtr, ytr, Xte, yte, cfg, rep_offset=0):
    return {
        "rbf_svm": {
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "predictions": [str(v) for v in yte],
        }
    }


def _patch_run(monkeypatch, tmp_path):
    monkeypatch.setattr(RE, "ROOT", tmp_path)
    (tmp_path / "results" / "tables").mkdir(parents=True, exist_ok=True)
    raw = _raw_frame()
    monkeypatch.setattr(RE, "run_raw_embeddings", lambda cfg: raw)
    monkeypatch.setattr(
        RE, "fit_pca_on_train", lambda Xtr, Xall, n, s: (Xtr[:, :n], Xall[:, :n], 99.0)
    )
    monkeypatch.setattr(
        RE, "select_features", lambda arm, X, y, k, seed, qubo: (list(range(k)), {})
    )
    monkeypatch.setattr(RE, "run_all_classifiers", _fake_classifiers)
    return raw


# --------------------------------------------------------------------------- #
# Q07 — content-aware fingerprint + config-keyed legacy feature cache
# --------------------------------------------------------------------------- #


def test_crops_content_fingerprint_detects_byte_change(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    img = d / "same.jpg"
    img.write_bytes(b"version one")

    name_before = F.crops_fingerprint(tmp_path, ["a"])
    content_before = F.crops_content_fingerprint(tmp_path, ["a"])

    img.write_bytes(b"different bytes, same filename")

    # content identity changes; the name-only validator mirror does not
    assert F.crops_content_fingerprint(tmp_path, ["a"]) != content_before
    assert F.crops_fingerprint(tmp_path, ["a"]) == name_before
    # identical content is stable (no spurious misses)
    assert F.crops_content_fingerprint(tmp_path, ["a"]) == F.crops_content_fingerprint(
        tmp_path, ["a"]
    )
    # class ordering is part of the identity
    assert F.crops_content_fingerprint(tmp_path, ["a"]) != F.crops_content_fingerprint(
        tmp_path, ["b", "a"]
    )


def test_run_features_cache_invalidates_on_content_and_config(tmp_path, monkeypatch):
    calls: list[int] = []

    class Model:
        def predict(self, imgs, verbose=0):
            calls.append(len(imgs))
            return np.ones((len(imgs), 6), dtype=np.float32)

    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F, "load_model", Model)
    monkeypatch.setattr(
        F.tf.keras.utils, "load_img", lambda *a, **kw: np.zeros((8, 8, 3), dtype=np.float32)
    )
    crops = tmp_path / "crops" / "a"
    crops.mkdir(parents=True)
    for i in range(6):
        (crops / f"{i}.jpg").write_bytes(b"fixture")

    cfg = {
        "seed": 1,
        "data": {"classes": ["a"], "crops_dir_override": "crops"},
        "features": {"backbone": "MobileNetV2", "pca_components": 2},
    }
    F.run_features(cfg)
    assert calls, "first call must compute embeddings"
    meta = json.loads((tmp_path / "data/features/features_a_meta.json").read_text())
    assert meta["features_config"] == {"backbone": "MobileNetV2", "pca_components": 2}
    assert "crops_content_fingerprint" in meta

    calls.clear()
    F.run_features(cfg)
    assert calls == [], "identical crops/config must be a cache hit"

    # config change (PCA budget) invalidates the cached CSV
    cfg3 = copy.deepcopy(cfg)
    cfg3["features"]["pca_components"] = 3
    F.run_features(cfg3)
    assert calls, "PCA-budget change must invalidate the cache"

    calls.clear()
    F.run_features(cfg3)
    assert calls == []
    # byte-level correction invalidates too
    (crops / "0.jpg").write_bytes(b"corrected")
    F.run_features(cfg3)
    assert calls, "content change must invalidate the cache"


def test_raw_cache_meta_stores_content_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "ROOT", tmp_path)
    crops = tmp_path / "crops" / "a"
    crops.mkdir(parents=True)
    for i in range(3):
        (crops / f"{i}.jpg").write_bytes(b"fixture")

    class Model:
        def predict(self, imgs, verbose=0):
            return np.ones((len(imgs), 2), dtype=np.float32)

    monkeypatch.setattr(F, "load_model", Model)
    monkeypatch.setattr(
        F.tf.keras.utils, "load_img", lambda *a, **kw: np.zeros((8, 8, 3), dtype=np.float32)
    )
    cfg = {"data": {"classes": ["a"], "crops_dir_override": "crops"}}
    F.run_raw_embeddings(cfg)
    meta = json.loads((tmp_path / "data/features/raw_a_meta.json").read_text())
    assert meta["crops_content_fingerprint"]["n_crops"] == 3
    assert "content_hash" in meta["crops_content_fingerprint"]


# --------------------------------------------------------------------------- #
# Q17 — unknown config keys are rejected; KNOWN_KEYS stays in sync
# --------------------------------------------------------------------------- #


def test_load_config_rejects_unknown_keys(tmp_path, monkeypatch):
    import yaml

    from src.quobo import config as C

    monkeypatch.setattr(C, "ROOT", tmp_path)
    good = _schema_cfg()
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump(good), encoding="utf-8")
    assert C.load_config("ok.yaml")["seed"] == 7

    (tmp_path / "top.yaml").write_text(yaml.safe_dump(dict(good, typo=1)), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level"):
        C.load_config("top.yaml")

    nested = copy.deepcopy(good)
    nested["qsvm"]["repz"] = 2
    (tmp_path / "sub.yaml").write_text(yaml.safe_dump(nested), encoding="utf-8")
    with pytest.raises(ValueError, match="qsvm"):
        C.load_config("sub.yaml")

    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        C.load_config("empty.yaml")


def test_validate_config_rejects_unknown_keys_and_cross_field():
    cfg = _schema_cfg()
    validate_config(cfg)  # baseline: valid

    unknown_top = dict(cfg, typo=1)
    with pytest.raises(ValueError, match="invalid experiment config"):
        validate_config(unknown_top)

    unknown_nested = copy.deepcopy(cfg)
    unknown_nested["qsvm"]["repz"] = 2
    with pytest.raises(ValueError, match="invalid experiment config"):
        validate_config(unknown_nested)

    # Q19: k must not exceed the candidate PCA pool
    cross = copy.deepcopy(cfg)
    cross["selection"]["k"] = 50
    cross["features"]["pca_components"] = 10
    with pytest.raises(ValueError, match="selection.k"):
        validate_config(cross)


def test_known_key_sets_match_schema():
    from src.quobo.config_schema import (
        DataConfig,
        ExperimentConfig,
        FeaturesConfig,
        QSVMConfig,
        QUBOConfig,
        QuoboConfig,
        SelectionConfig,
    )

    assert set(QuoboConfig.model_fields) == KNOWN_KEYS
    models = {
        "data": DataConfig,
        "features": FeaturesConfig,
        "selection": SelectionConfig,
        "qubo": QUBOConfig,
        "qsvm": QSVMConfig,
        "experiment": ExperimentConfig,
    }
    for section, model in models.items():
        assert set(model.model_fields) == KNOWN_SUBKEYS[section], section


def test_shipped_configs_pass_strict_load_and_schema():
    for path in (
        "configs/experiment.yaml",
        "configs/experiment_6class.yaml",
        "configs/experiment_trashnet.yaml",
    ):
        cfg = load_config(path)
        assert validate_config(cfg) is not None


# --------------------------------------------------------------------------- #
# Runner consumes the validated model; Z_full label tracks pca_components
# --------------------------------------------------------------------------- #


def test_run_uses_validated_normalized_config(tmp_path, monkeypatch):
    _patch_run(monkeypatch, tmp_path)
    captured = {}

    def capture(cfg):
        captured["cfg"] = copy.deepcopy(cfg)
        return _raw_frame()

    monkeypatch.setattr(RE, "run_raw_embeddings", capture)

    cfg = _schema_cfg()
    assert "parallel" not in cfg["qubo"] and "early_stop" not in cfg["experiment"]
    RE.run(cfg, run_id="run_norm")

    effective = captured["cfg"]
    assert effective["qubo"]["parallel"] is False  # schema default in effect
    assert effective["experiment"]["early_stop"] is False
    assert effective["experiment"]["track_mlflow"] is False
    assert effective["qsvm"]["max_qsvm_features"] == 20
    # the caller's dict is not mutated
    assert "parallel" not in cfg["qubo"]


def test_z_full_label_tracks_pca_components(tmp_path, monkeypatch):
    _patch_run(monkeypatch, tmp_path)
    cfg = _schema_cfg()
    cfg["features"]["pca_components"] = 10
    cfg["experiment"]["arms"] = ["Z_full"]

    results = RE.run(cfg, run_id="run_full")

    assert "Z_full" not in RE.ARM_LABELS  # no hardcoded Full-50
    assert RE.arm_label("Z_full", 10) == "Full-10"
    assert set(results["arm"]) == {"Full-10"}
    payload = json.loads(
        (tmp_path / "experiments" / "run_full" / "rep0_Z_full.json").read_text(encoding="utf-8")
    )
    assert payload["arm"] == "Z_full"  # provenance key unchanged
    assert payload["arm_label"] == "Full-10"  # label consistent with rows


# --------------------------------------------------------------------------- #
# QSVM memory guard (k > max_qsvm_features skips the statevector QSVM)
# --------------------------------------------------------------------------- #


def test_run_all_classifiers_skips_qsvm_above_feature_cap(caplog):
    from src.quobo.qsvm import run_all_classifiers

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 25))
    y = (X[:, 0] > 0).astype(int)
    cfg = {
        "seed": 0,
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 50,
            "test_fraction": 0.25,
            "tune_rbf": False,
            "max_qsvm_features": 20,
        },
    }
    with caplog.at_level("WARNING"):
        metrics = run_all_classifiers(X[:45], y[:45], X[45:], y[45:], cfg, rep_offset=0)

    assert "qsvm" not in metrics
    assert "rbf_svm" in metrics and "linear_svm" in metrics
    assert "skipping QSVM" in caplog.text


# --------------------------------------------------------------------------- #
# Q15 — the parallel coarse scan actually submits futures and stays deterministic
# --------------------------------------------------------------------------- #


def test_parallel_qubo_submits_futures(monkeypatch):
    import concurrent.futures as cf

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 20))
    y = (X[:, 0] > 0).astype(int)

    real_submit = cf.ThreadPoolExecutor.submit
    submitted: list[float] = []

    def spy(self, fn, *args, **kwargs):
        submitted.append(args[0])
        return real_submit(self, fn, *args, **kwargs)

    monkeypatch.setattr(cf.ThreadPoolExecutor, "submit", spy)

    sel, info = select_qubo(X, y, 4, n_bins=8, sweeps=1000, repeats=5, seed=42, parallel=True)

    assert len(submitted) == 10, "all coarse alphas must be submitted concurrently"
    assert len(sel) == 4 and all(0 <= i < 20 for i in sel)
    assert info["parallel"] is True
    assert 0.0 <= info["alpha"] <= 1.0

    sel2, info2 = select_qubo(X, y, 4, n_bins=8, sweeps=1000, repeats=5, seed=42, parallel=True)
    assert sel2 == sel and info2["alpha"] == info["alpha"]

    sel_seq, info_seq = select_qubo(
        X, y, 4, n_bins=8, sweeps=1000, repeats=5, seed=42, parallel=False
    )
    assert len(sel_seq) == 4 and info_seq["parallel"] is False


# --------------------------------------------------------------------------- #
# Q24 — MLflow store override via QUOBO_MLFLOW_TRACKING_URI
# --------------------------------------------------------------------------- #


def test_log_to_mlflow_tracking_uri_env_override(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from src.quobo.run_experiment import log_to_mlflow

    fake = MagicMock()
    fake.active_run.return_value.info.run_id = "run-123"
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    cfg = {
        "seed": 42,
        "data": {"classes": ["a", "b"], "min_images_per_class": 40},
        "features": {"backbone": "MobileNetV2", "pca_components": 50},
        "selection": {"k": 8},
        "qubo": {"mi_bins": 16, "sa_sweeps": 20000, "sa_repeats": 10, "parallel": False},
        "qsvm": {"reps": 1, "max_train_samples": 400, "test_fraction": 0.25},
        "experiment": {"n_repeats": 25, "arms": ["D_qubo"]},
    }
    results = pd.DataFrame(
        [
            {
                "repeat": 0,
                "arm": "QUBO-8",
                "classifier": "rbf_svm",
                "accuracy": 0.40,
                "macro_f1": 0.38,
            },
        ]
    )
    results.to_csv(tmp_path / "results_all.csv", index=False)

    monkeypatch.setenv("QUOBO_MLFLOW_TRACKING_URI", "sqlite:///durable/quobo.sqlite")
    assert log_to_mlflow(results, cfg, tmp_path) is True
    fake.set_tracking_uri.assert_called_once_with("sqlite:///durable/quobo.sqlite")

    fake.reset_mock()
    fake.active_run.return_value.info.run_id = "run-456"
    monkeypatch.delenv("QUOBO_MLFLOW_TRACKING_URI", raising=False)
    assert log_to_mlflow(results, cfg, tmp_path) is True
    fallback = fake.set_tracking_uri.call_args[0][0]
    assert fallback.startswith("sqlite:///") and "quobo_mlruns.sqlite" in fallback
