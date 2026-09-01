"""Stage 4: classifiers — QSVC (quantum kernel SVM) + classical baselines.

QSVM: ZZFeatureMap(k qubits) -> FidelityQuantumKernel -> QSVC (sklearn SVC subclass).
Baselines: RBF-SVM on the same k features (mandatory per Huang et al. 2021),
plus linear SVM for reference.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

log = logging.getLogger(__name__)


def make_qsvc(k: int, cfg: dict):
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    # function form (Qiskit >=2.1): plain QuantumCircuit, not deprecated BlueprintCircuit.
    # Statevector kernel: one statevector per sample + classical fidelities —
    # orders of magnitude faster than FidelityQuantumKernel's shot-based pairwise
    # circuit executions on a noiseless simulator, with identical ideal results.
    fm = zz_feature_map(
        feature_dimension=k,
        reps=cfg["qsvm"]["reps"],
        entanglement=cfg["qsvm"]["entanglement"],
    )
    kernel = FidelityStatevectorKernel(feature_map=fm, enforce_psd=True)
    return QSVC(quantum_kernel=kernel), kernel


def evaluate(model, Xtr, ytr, Xte, yte) -> dict:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Xte)
    pred_s = time.perf_counter() - t0
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "fit_seconds": fit_s,
        "predict_seconds": pred_s,
        "predictions": [str(p) for p in pred],  # JSON-safe
    }


def fit_scale_for_feature_map(Xtr: np.ndarray, Xte: np.ndarray) -> tuple:
    """Fit the angle scaler on TRAIN only, transform both sets identically.

    Returns (Xtr_scaled, Xte_scaled). Range [0.05, 1.52] stays inside the
    ZZFeatureMap's 2*pi periodicity with margin (raw PCA scores are unbounded;
    unbounded angles alias and the kernel collapses).
    """
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0.05, 1.52)).fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte)


def run_all_classifiers(Xtr, ytr, Xte, yte, cfg: dict, rep_offset: int = 0) -> dict:
    """Train QSVC + classical baselines on the SAME features. Returns metrics dict.

    rep_offset varies the training-subsample seed per repeat (same protocol
    across every caller — the main run and any downstream analysis).
    The statevector QSVM is physically limited to ~20 qubits (2^n memory);
    for wider feature sets (e.g. the Z_full 50-feature ceiling) it is skipped
    and only classical baselines run."""
    k = Xtr.shape[1]
    results = {}
    run_qsvm = k <= cfg.get("qsvm", {}).get("max_qsvm_features", 20)

    max_n = cfg["qsvm"]["max_train_samples"]
    if len(ytr) > max_n:
        rng = np.random.default_rng(cfg["seed"] + rep_offset)
        idx = rng.choice(len(ytr), size=max_n, replace=False)
        Xtr_c, ytr_c = np.asarray(Xtr)[idx], np.asarray(ytr)[idx]
    else:
        idx = None
        Xtr_c, ytr_c = Xtr, np.asarray(ytr)

    # QSVM (angle-scaled inputs; scaler fit on train only)
    if run_qsvm:
        Xtr_q, Xte_q = fit_scale_for_feature_map(
            np.asarray(Xtr, dtype=float)[idx] if idx is not None else np.asarray(Xtr, dtype=float),
            np.asarray(Xte, dtype=float),
        )
        ytr_s = ytr_c
        qsvc, _ = make_qsvc(k, cfg)
        qsvc.class_weight = "balanced"
        results["qsvm"] = evaluate(qsvc, Xtr_q, ytr_s, Xte_q, yte)
    else:
        log.warning("skipping QSVM at k=%d features (> max_qsvm_features) — classical arms only", k)

    # classical RBF-SVM (same subsample for fairness)
    results["rbf_svm"] = evaluate(
        SVC(kernel="rbf", class_weight="balanced", random_state=cfg["seed"]),
        Xtr_c, ytr_c, Xte, yte,
    )
    # tuned RBF-SVM: small grid over C/gamma on the SAME training subsample.
    # Neutralizes the 'weak classical baseline' objection (Huang 2021) — the
    # grid is seconds on 400x8 data.
    if cfg.get("qsvm", {}).get("tune_rbf", False):
        from sklearn.model_selection import GridSearchCV

        grid = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced"),
            {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1]},
            cv=3, n_jobs=1,
        )
        grid.fit(Xtr_c, ytr_c)
        results["rbf_svm_tuned"] = evaluate(grid.best_estimator_, Xtr_c, ytr_c, Xte, yte)
    # linear SVM reference
    results["linear_svm"] = evaluate(
        SVC(kernel="linear", class_weight="balanced", random_state=cfg["seed"]),
        Xtr_c, ytr_c, Xte, yte,
    )
    return results
