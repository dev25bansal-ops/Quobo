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
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    from qiskit_machine_learning.algorithms import QSVC

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


def run_all_classifiers(Xtr, ytr, Xte, yte, cfg: dict) -> dict:
    """Train QSVC + classical baselines on the SAME features. Returns metrics dict."""
    k = Xtr.shape[1]
    results = {}

    # QSVM gets angle-scaled inputs (scaler fit on train only); classical
    # SVMs get the raw columns
    Xtr_q, Xte_q = fit_scale_for_feature_map(
        np.asarray(Xtr, dtype=float), np.asarray(Xte, dtype=float)
    )

    max_n = cfg["qsvm"]["max_train_samples"]
    if len(ytr) > max_n:
        rng = np.random.default_rng(cfg["seed"] + cfg.get("_rep_offset", 0))
        idx = rng.choice(len(ytr), size=max_n, replace=False)
        Xtr_s, ytr_s = Xtr_q[idx], np.asarray(ytr)[idx]
        Xtr_c, ytr_c = np.asarray(Xtr)[idx], np.asarray(ytr)[idx]
    else:
        Xtr_s, ytr_s = Xtr_q, np.asarray(ytr)
        Xtr_c, ytr_c = Xtr, np.asarray(ytr)

    # QSVM
    qsvc, _ = make_qsvc(k, cfg)
    qsvc.class_weight = "balanced"
    results["qsvm"] = evaluate(qsvc, Xtr_s, ytr_s, Xte_q, yte)

    # classical RBF-SVM (same subsample for fairness)
    results["rbf_svm"] = evaluate(
        SVC(kernel="rbf", class_weight="balanced", random_state=cfg["seed"]),
        Xtr_c, ytr_c, Xte, yte,
    )
    # linear SVM reference
    results["linear_svm"] = evaluate(
        SVC(kernel="linear", class_weight="balanced", random_state=cfg["seed"]),
        Xtr_c, ytr_c, Xte, yte,
    )
    return results
