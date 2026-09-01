"""Smoke test: end-to-end pipeline on synthetic data (no TACO download needed).

Verifies: selection arms run, QSVM circuit builds, kernel evaluates, all
classifiers train/predict. Small sizes so it finishes in a couple of minutes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = ["smoke", "configs/experiment.yaml"]
from src.quobo.config import load_config
from src.quobo.selection import select_features
from src.quobo.qsvm import run_all_classifiers

cfg = load_config()
rng = np.random.default_rng(cfg["seed"])

# synthetic 6-class problem, 50 candidate features, only 6 informative + correlated noise
n, d, k = 240, 50, 8
centers = rng.normal(size=(6, 6))
y = rng.integers(0, 6, n)
X_info = centers[y] + rng.normal(scale=0.7, size=(n, 6))
noise = rng.normal(size=(n, d - 6))
corr = noise[:, :6] * 0.9 + X_info * 0.1  # redundant copies of informative features
X = np.hstack([X_info, corr, noise[:, 6:]])

print(f"synthetic: X{X.shape}, classes={len(set(y))}")
for arm in cfg["experiment"]["arms"]:
    sel, info = select_features(arm, X, y, k, cfg["seed"], cfg["qubo"])
    print(f"{arm}: selected={sel}")
    assert len(sel) == k, f"{arm} selected {len(sel)} != {k}"

# QSVM + baselines on QUBO-selected features
from src.quobo.selection import select_qubo
sel, _ = select_qubo(X, y, k, n_bins=8, sweeps=2000, repeats=5)
Xtr, Xte, ytr, yte = (
    X[:180][:, sel], X[180:][:, sel], y[:180], y[180:]
)
cfg_small = dict(cfg)
cfg_small["qsvm"] = dict(cfg["qsvm"], max_train_samples=120)
metrics = run_all_classifiers(Xtr, ytr, Xte, yte, cfg_small)
for clf, m in metrics.items():
    print(f"{clf}: acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f} fit={m['fit_seconds']:.1f}s")
print("SMOKE TEST PASSED")
