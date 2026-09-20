"""Unit tests for QSVM scaling and the feature-selection arms.

Covers: scaler fit-on-train-only (QML-001), selection ground-truth recovery,
selection determinism, and the parallel QUBO alpha search (Enhancement 2).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quobo.qsvm import fit_scale_for_feature_map


# ---------- QML-001: scaler fit on train only ----------


def test_scaler_params_from_train_only():
    Xtr = np.array([[10.0], [20.0], [30.0]])
    Xte = np.array([[5.0], [40.0]])  # outside train bounds
    tr_s, te_s = fit_scale_for_feature_map(Xtr, Xte)
    # scaler range is fixed 0.05-1.52: train min->0.05, max->1.52
    assert tr_s[0, 0] == pytest.approx(0.05)
    assert tr_s[2, 0] == pytest.approx(1.52)
    # test values map OUTSIDE the range (no clip, no test-stat fitting)
    assert te_s[0, 0] < 0.05
    assert te_s[1, 0] > 1.52


def test_scaler_identical_transform_of_same_point():
    Xtr = np.random.default_rng(0).normal(size=(50, 4))
    Xte = Xtr[:5].copy()
    tr_s, te_s = fit_scale_for_feature_map(Xtr, Xte)
    # a train row appearing in 'test' must map to the same angles
    np.testing.assert_allclose(tr_s[:5], te_s)


# ---------- selection ground-truth recovery ----------


def test_selection_recovers_informative_features():
    from src.quobo.selection import select_features

    rng = np.random.default_rng(42)
    n = 50
    informative = [3, 11, 24, 37, 45]
    X = rng.normal(size=(600, n))
    for i, c in enumerate(informative):
        X[:, c] += (i - 2) * 2.0 * np.where(np.arange(600) % 6 == 0, 1.0, -1.0)
    y = (np.arange(600) % 6 == 0).astype(int)
    cfg = {"mi_bins": 16, "sa_sweeps": 2000, "sa_repeats": 5}
    for arm in ("C_mi", "D_qubo"):
        sel, _ = select_features(arm, X, y, 5, 42, cfg)
        overlap = len(set(sel) & set(informative))
        assert overlap >= 3, f"{arm} recovered only {overlap}/5 informative features: {sel}"


# ---------- reproducibility ----------


def test_selection_deterministic():
    from src.quobo.selection import select_features

    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 30))
    y = (X[:, 0] > 0).astype(int)
    cfg = {"mi_bins": 16, "sa_sweeps": 2000, "sa_repeats": 5}
    s1, _ = select_features("D_qubo", X, y, 4, 42, cfg)
    s2, _ = select_features("D_qubo", X, y, 4, 42, cfg)
    assert s1 == s2


# ---------- Enhancement 2: parallel QUBO alpha search ----------


def test_parallel_qubo_returns_k_and_matches_mode():
    from src.quobo.selection import select_qubo

    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 30))
    y = (X[:, 0] > 0).astype(int)
    cfg = {"n_bins": 16, "sweeps": 2000, "repeats": 5, "seed": 42}
    sel_seq, info_seq = select_qubo(X, y, 4, parallel=False, **cfg)
    sel_par, info_par = select_qubo(X, y, 4, parallel=True, **cfg)
    assert len(sel_seq) == 4 and len(sel_par) == 4
    assert info_seq["parallel"] is False and info_par["parallel"] is True
    # parallel mode must be deterministic too
    _, info_par2 = select_qubo(X, y, 4, parallel=True, **cfg)
    assert info_par2["alpha"] == info_par["alpha"]


def test_qubo_handles_string_labels():
    """Regression (M-09 re-run): run() forwards string labels from the raw
    ``label`` column; compute_mi_table/select_qubo must encode them without
    crashing and yield the same selection as an integer-labeled call (MI is
    invariant under any bijective label map)."""
    from src.quobo.selection import compute_mi_table, select_qubo

    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 12))
    labels = np.array(["a", "b", "c"] * 100)
    X[:, 2] += (labels == "c") * 3.0
    X[:, 5] += (labels == "a") * 2.0
    sel_str, info_str = select_qubo(X, labels, 4, n_bins=8, sweeps=1000, repeats=5, seed=42)
    _, y_int = np.unique(labels, return_inverse=True)
    sel_int, _ = select_qubo(X, y_int, 4, n_bins=8, sweeps=1000, repeats=5, seed=42)
    assert info_str["k_selected"] == 4
    assert sel_str == sel_int
    Is, Rs = compute_mi_table(X, labels, 8)
    Ii, Ri = compute_mi_table(X, y_int, 8)
    assert np.allclose(Is, Ii) and np.allclose(Rs, Ri)
