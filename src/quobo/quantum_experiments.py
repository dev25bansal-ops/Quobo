"""Bounded, local quantum-kernel experiments and an opt-in SamplerV2 harness.

No module import performs I/O, credential discovery, or remote execution. Inputs are
explicit, group-disjoint train/test splits; all data-dependent transforms use train.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "quantum-experiments-v1"


def package_versions() -> dict[str, str]:
    result = {}
    for name in (
        "numpy",
        "scikit-learn",
        "qiskit",
        "qiskit-machine-learning",
        "qiskit-ibm-runtime",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _matrix(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite 2D matrix")
    return value


def _canonical_ids(value: np.ndarray) -> np.ndarray:
    """Canonical string form for identifiers so cross-split types cannot hide
    collisions: integer 0 (train) and float 0.0 (test) must collide, not pass."""
    value = np.asarray(value)
    if value.dtype.kind in "biu":
        return np.array([str(int(v)) for v in value], dtype=str)
    if value.dtype.kind == "f":

        def one(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else repr(float(v))

        return np.array([one(v) for v in value], dtype=str)
    return value.astype(str)


@dataclass
class FeatureSplit:
    """Explicit split with original-source groups, unique sample IDs and provenance.

    ``transform_fit`` must be ``raw`` or ``train_only``. This is an input contract,
    not a way to detect undisclosed upstream leakage. Groups identify source images,
    not crop IDs; sibling crops must share a group.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray
    ids_train: np.ndarray
    ids_test: np.ndarray
    transform_fit: str = "raw"

    def __post_init__(self):
        self.X_train = _matrix(self.X_train, "X_train")
        self.X_test = _matrix(self.X_test, "X_test")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("train/test feature dimensions differ")
        if self.transform_fit not in {"raw", "train_only"}:
            raise ValueError("transform_fit must declare raw or train_only preprocessing")
        for side in ("train", "test"):
            for prefix in ("y", "groups", "ids"):
                name = f"{prefix}_{side}"
                value = np.asarray(getattr(self, name))
                if value.ndim != 1 or len(value) != len(getattr(self, f"X_{side}")):
                    raise ValueError(f"{name} must be 1D and match X_{side}")
                if value.dtype.kind not in "biufUS" or (
                    value.dtype.kind in "f" and not np.isfinite(value).all()
                ):
                    raise ValueError(f"{name} must contain finite numeric or string identifiers")
                value = _canonical_ids(value)
                if np.any(value == ""):
                    raise ValueError(f"{name} contains empty identifiers")
                setattr(self, name, value)
        if set(self.groups_train) & set(self.groups_test):
            raise ValueError("source groups overlap across train/test (data leakage)")
        ids = np.concatenate([self.ids_train, self.ids_test])
        if len(np.unique(ids)) != len(ids):
            raise ValueError("sample IDs must be unique across both splits")
        if len(np.unique(self.y_train)) < 2:
            raise ValueError("training requires at least two classes")
        if not set(self.y_test) <= set(self.y_train):
            raise ValueError("test contains labels absent from training")

    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.transform_fit.encode())
        for name in (
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "groups_train",
            "groups_test",
            "ids_train",
            "ids_test",
        ):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


def load_split(path: str | Path) -> FeatureSplit:
    """Read only a pickle-free NPZ split, never a globally transformed feature CSV."""
    with np.load(path, allow_pickle=False) as data:
        required = (
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "groups_train",
            "groups_test",
            "ids_train",
            "ids_test",
            "transform_fit",
        )
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"split NPZ missing keys: {sorted(missing)}")
        values = {name: data[name] for name in required}
        values["transform_fit"] = str(values["transform_fit"].item())
    return FeatureSplit(**values)


def smoke_split(seed: int = 42, features: int = 12) -> FeatureSplit:
    """Small synthetic fixture, not TACO and not a scientific performance result."""
    rng = np.random.default_rng(seed)
    y_train = np.tile(np.array(["a", "b", "c"]), 4)
    y_test = np.tile(np.array(["a", "b", "c"]), 2)
    X_train = rng.normal(size=(len(y_train), features))
    X_test = rng.normal(size=(len(y_test), features))
    for X, y in ((X_train, y_train), (X_test, y_test)):
        X[:, :2] += np.searchsorted(["a", "b", "c"], y)[:, None]
    return FeatureSplit(
        X_train,
        y_train,
        X_test,
        y_test,
        np.array([f"train-source-{i // 2}" for i in range(len(y_train))]),
        np.array([f"test-source-{i // 2}" for i in range(len(y_test))]),
        np.array([f"train-{i}" for i in range(len(y_train))]),
        np.array([f"test-{i}" for i in range(len(y_test))]),
    )


def _sample_indices(labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Seeded cap retaining every train class without consulting held-out labels."""
    classes, counts = np.unique(labels, return_counts=True)
    if cap < 2 * len(classes):
        raise ValueError("train sample cap must be at least twice the number of classes")
    if np.any(counts < 2):
        raise ValueError("MI selection requires at least two training rows per class")
    if len(labels) <= cap:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    required = np.concatenate(
        [rng.choice(np.flatnonzero(labels == label), 2, replace=False) for label in classes]
    )
    remaining = np.setdiff1d(np.arange(len(labels)), required)
    chosen = rng.choice(remaining, cap - len(required), replace=False)
    return np.sort(np.concatenate([required, chosen]))


def prepare_split(
    split: FeatureSplit, budget: int, seed: int = 42, max_train: int = 48, max_test: int = 64
) -> tuple[dict, dict]:
    """Train-only MI ranking and angle scaling on the capped training population.

    Evaluation capping is random and label-independent. IDs/groups are retained so
    downstream consumers can reconstruct the exact evaluation population.
    """
    if not 2 <= budget <= min(12, split.X_train.shape[1]):
        raise ValueError("feature budget must be 2..12 and not exceed input width")
    if not 2 <= max_train <= 128 or not 1 <= max_test <= 256:
        raise ValueError("caps must be train 2..128 and test 1..256")
    train_idx = _sample_indices(split.y_train, max_train, seed)
    test_idx = np.sort(
        np.random.default_rng(seed + 1).choice(
            len(split.y_test), min(max_test, len(split.y_test)), replace=False
        )
    )
    Xtr = split.X_train[train_idx]
    # MI handles the label strings as categories, never as ordinal regression targets.
    scores = mutual_info_classif(
        Xtr, split.y_train[train_idx], random_state=seed, n_neighbors=min(3, max(1, len(Xtr) - 1))
    )
    selected = np.argsort(-scores, kind="stable")[:budget]
    scaler = MinMaxScaler(feature_range=(0.05, 1.52)).fit(Xtr[:, selected])
    prepared = {
        "X_train": scaler.transform(Xtr[:, selected]),
        "y_train": split.y_train[train_idx],
        "X_test": scaler.transform(split.X_test[test_idx][:, selected]),
        "y_test": split.y_test[test_idx],
    }
    provenance = {
        "split_sha256": split.fingerprint(),
        "transform_fit": split.transform_fit,
        "selection": "mutual_info_classif_train_only",
        "selected": selected.tolist(),
        "mi_scores": scores.tolist(),
        "angle_range": [0.05, 1.52],
        "scaler_data_min": scaler.data_min_.tolist(),
        "scaler_data_max": scaler.data_max_.tolist(),
        "scaler_clip": False,
        "seed": seed,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "ids_train": split.ids_train[train_idx].tolist(),
        "ids_test": split.ids_test[test_idx].tolist(),
        "groups_train": split.groups_train[train_idx].tolist(),
        "groups_test": split.groups_test[test_idx].tolist(),
    }
    return prepared, provenance


def _circuit_api():
    try:
        from qiskit.circuit.library import zz_feature_map
    except ImportError as exc:
        raise RuntimeError("Install qiskit>=2.1 to build quantum circuits") from exc
    return zz_feature_map


def _validate_map(qubits: int, reps: int, entanglement: str):
    if not 2 <= qubits <= 12 or reps not in {1, 2, 3}:
        raise ValueError("quantum maps require 2..12 qubits and reps in {1,2,3}")
    if entanglement not in {"linear", "full"}:
        raise ValueError("entanglement must be linear or full")


def make_feature_map(
    qubits: int, reps: int = 1, entanglement: str = "linear", trainable: bool = False
):
    """ZZ encoding with optional per-feature multiplicative input weights.

    These weights occur *inside data encoding*, not in a final shared unitary
    (which would cancel in a fidelity). At weights=1 this is exactly fixed ZZ.
    """
    _validate_map(qubits, reps, entanglement)
    circuit = _circuit_api()(qubits, reps=reps, entanglement=entanglement)
    if not trainable:
        return circuit, ()
    from qiskit.circuit import ParameterVector

    theta = ParameterVector("theta", qubits)
    features = list(circuit.parameters)
    circuit = circuit.assign_parameters({x: x * t for x, t in zip(features, theta)})
    return circuit, tuple(theta)


def make_statevector_kernel(
    qubits: int, reps: int = 1, entanglement: str = "linear", trainable: bool = False
):
    try:
        from qiskit_machine_learning.kernels import (
            FidelityStatevectorKernel,
            TrainableFidelityStatevectorKernel,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install qiskit-machine-learning>=0.9 and qiskit>=2.1 "
            "for local statevector experiments"
        ) from exc
    circuit, theta = make_feature_map(qubits, reps, entanglement, trainable)
    if trainable:
        return TrainableFidelityStatevectorKernel(
            feature_map=circuit,
            training_parameters=theta,
            shots=None,
            auto_clear_cache=True,
            enforce_psd=True,
        )
    return FidelityStatevectorKernel(feature_map=circuit, shots=None, enforce_psd=True)


def centered_target_alignment(kernel: np.ndarray, labels: np.ndarray) -> float:
    """Frobenius alignment with centered one-hot class Gram (multiclass-safe).

    No ordinal label products: target[i,j] = 1 iff labels match. Centering removes
    the trivial constant component. This objective is not test accuracy.
    """
    K = _matrix(kernel, "kernel")
    labels = np.asarray(labels)
    if labels.ndim != 1 or K.shape != (len(labels), len(labels)):
        raise ValueError("kernel must be square and match one-dimensional labels")
    if len(np.unique(labels)) < 2:
        raise ValueError("alignment needs at least two classes")
    target = (labels[:, None] == labels[None, :]).astype(float)

    def center(matrix):
        return matrix - matrix.mean(0) - matrix.mean(1)[:, None] + matrix.mean()

    K, target = center(K), center(target)
    norm = np.linalg.norm(K) * np.linalg.norm(target)
    return float(np.sum(K * target) / norm) if norm > 1e-14 else 0.0


def train_alignment(
    kernel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int = 42,
    iterations: int = 10,
    learning_rate: float = 0.2,
    perturbation: float = 0.1,
) -> dict:
    """Explicit first-order SPSA, bounded [0.1,3] weights; restore best evaluation.

    This is a local implementation of SPSA, not QML's SVC-loss trainer. At most
    1+3*iterations kernel evaluations, all on train. Include initial weights among
    checkpoints so the restored training objective cannot be worse than fixed ZZ.
    """
    X = _matrix(X_train, "X_train")
    y = np.asarray(y_train)
    if len(X) > 128 or not 0 <= iterations <= 100:
        raise ValueError("training bounded to 128 samples and 0..100 iterations")
    if (
        not np.isfinite([learning_rate, perturbation]).all()
        or min(learning_rate, perturbation) <= 0
    ):
        raise ValueError("SPSA rates must be positive and finite")
    point = np.ones(len(kernel.training_parameters))
    if len(point) == 0:
        raise ValueError("kernel has no trainable parameters")
    rng = np.random.default_rng(seed)
    best_point = point.copy()
    best_loss = float("inf")
    history = []

    def objective(parameters, iteration, role):
        nonlocal best_point, best_loss
        parameters = np.clip(parameters, 0.1, 3.0)
        kernel.assign_training_parameters(parameters)
        loss = -centered_target_alignment(kernel.evaluate(X), y)
        if not np.isfinite(loss):
            raise ValueError("nonfinite kernel objective")
        history.append(
            {"iteration": iteration, "role": role, "loss": loss, "parameters": parameters.tolist()}
        )
        if loss < best_loss:
            best_loss, best_point = loss, parameters.copy()
        return loss

    initial_loss = objective(point, 0, "initial")
    for iteration in range(1, iterations + 1):
        delta = rng.choice([-1.0, 1.0], len(point))
        ck = perturbation / iteration**0.101
        ak = learning_rate / iteration**0.602
        plus = objective(point + ck * delta, iteration, "plus")
        minus = objective(point - ck * delta, iteration, "minus")
        gradient = (plus - minus) / (2 * ck) * delta
        point = np.clip(point - ak * gradient, 0.1, 3.0)
        objective(point, iteration, "iterate")
    kernel.assign_training_parameters(best_point)
    return {
        "optimizer": "explicit-first-order-SPSA",
        "objective": "centered-target-alignment",
        "seed": seed,
        "iterations": iterations,
        "evaluations": len(history),
        "learning_rate": learning_rate,
        "perturbation": perturbation,
        "bounds": [0.1, 3.0],
        "initial_parameters": np.ones(len(point)).tolist(),
        "best_parameters": best_point.tolist(),
        "initial_alignment": -initial_loss,
        "best_alignment": -best_loss,
        "best_checkpoint_restored": True,
        "history": history,
    }


def evaluate_kernel(kernel, prepared: dict, C: float = 1.0) -> dict:
    if not np.isfinite(C) or C <= 0:
        raise ValueError("SVC C must be positive and finite")
    Xtr, Xte = prepared["X_train"], prepared["X_test"]
    Ktr = kernel.evaluate(Xtr)
    Kte = kernel.evaluate(Xte, Xtr)
    model = SVC(kernel="precomputed", C=C, class_weight="balanced")
    model.fit(Ktr, prepared["y_train"])
    predictions = model.predict(Kte)
    return {
        "accuracy": float(accuracy_score(prepared["y_test"], predictions)),
        "macro_f1": float(
            f1_score(
                prepared["y_test"],
                predictions,
                average="macro",
                labels=np.unique(prepared["y_train"]),
                zero_division=0,
            )
        ),
        "predictions": predictions.tolist(),
        "y_test": prepared["y_test"].tolist(),
        "classes": model.classes_.tolist(),
        "C": C,
        "C_tuned": False,
        "train_alignment": centered_target_alignment(Ktr, prepared["y_train"]),
    }


def train_kernel_experiment(
    split: FeatureSplit,
    *,
    budget: int = 4,
    reps: int = 1,
    entanglement: str = "linear",
    seed: int = 42,
    iterations: int = 10,
    max_train: int = 48,
    max_test: int = 64,
) -> dict:
    prepared, provenance = prepare_split(split, budget, seed, max_train, max_test)
    fixed = make_statevector_kernel(budget, reps, entanglement)
    trained = make_statevector_kernel(budget, reps, entanglement, trainable=True)
    training = train_alignment(
        trained, prepared["X_train"], prepared["y_train"], seed=seed, iterations=iterations
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": IMPLEMENTATION_VERSION,
        "mode": "local-exact-statevector",
        "packages": package_versions(),
        "map": {
            "family": "ZZ-per-feature-input-weights",
            "budget": budget,
            "reps": reps,
            "entanglement": entanglement,
        },
        "provenance": provenance,
        "training": training,
        "fixed": evaluate_kernel(fixed, prepared),
        "trained": evaluate_kernel(trained, prepared),
        "interpretation": "Held-out comparison, not evidence of quantum advantage; "
        "alignment improvement need not improve test accuracy.",
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def write_new_json(path: str | Path, value: Any):
    """Publish complete JSON atomically without replacing existing results."""
    import tempfile

    payload = _json_bytes(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link publishes a complete file exclusively.
        # Unlike replace/rename, it fails if another writer already published.
        os.link(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def ablation_grid(reps=(1, 2, 3), entanglements=("linear", "full"), budgets=(4, 6, 8, 12)):
    configs = []
    for r, e, b in itertools.product(reps, entanglements, budgets):
        _validate_map(b, r, e)
        if b not in {4, 6, 8, 12}:
            raise ValueError("ablation budgets must be one of 4,6,8,12")
        config = {"reps": r, "entanglement": e, "budget": b}
        if config not in configs:
            configs.append(config)
    if not configs:
        raise ValueError("ablation grid cannot be empty")
    return configs


def run_ablation(
    split: FeatureSplit,
    output: str | Path,
    *,
    configs=None,
    seed: int = 42,
    max_train: int = 48,
    max_test: int = 64,
    dry_run: bool = True,
    max_runs: int = 1,
) -> dict:
    """Plan or execute fixed-ZZ ablations; resume by input/config/version hash.

    A new manifest never overwrites another; existing keyed results are verified
    before reuse. Default plans 24 cells and executes none. Execute at most one
    missing cell by default, rather than accidentally launching a full sweep.
    """
    configs = ablation_grid() if configs is None else configs
    if not 1 <= max_runs <= 24:
        raise ValueError("max_runs must be 1..24")
    if not configs or len(configs) > 24:
        raise ValueError("provide 1..24 ablation configurations")
    for config in configs:
        ablation_grid([config["reps"]], [config["entanglement"]], [config["budget"]])
        if config["budget"] > split.X_train.shape[1]:
            raise ValueError("ablation budget exceeds input width")
    if not 2 <= max_train <= 128 or not 1 <= max_test <= 256:
        raise ValueError("caps must be train 2..128 and test 1..256")
    root = Path(output)
    protocol = {
        "implementation": IMPLEMENTATION_VERSION,
        "schema_version": SCHEMA_VERSION,
        "split_sha256": split.fingerprint(),
        "seed": seed,
        "max_train": max_train,
        "max_test": max_test,
        "packages": package_versions(),
        "selection": "mutual_info_classif_train_only",
        "model": "fixed-ZZ-SVC-C1",
    }
    cells = []
    for config in configs:
        specification = {**protocol, "config": config}
        key = hashlib.sha256(_json_bytes(specification)).hexdigest()
        cells.append({"key": key, "specification": specification, "result_file": f"{key}.json"})
    manifest = {
        "protocol": protocol,
        "cells": cells,
        "estimated_statevectors": sum(
            2 * min(max_train, len(split.y_train)) + min(max_test, len(split.y_test)) for _ in cells
        ),
        "note": "Reps are circuit depth, not statistical repetitions. "
        "No cross-cell winner is selected using held-out scores.",
    }
    manifest_key = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    manifest_path = root / f"manifest-{manifest_key}.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("existing ablation manifest is inconsistent")
    else:
        write_new_json(manifest_path, manifest)
    completed, reused = [], []
    for cell in cells:
        target = root / cell["result_file"]
        if target.exists():
            data = json.loads(target.read_text())
            if data.get("key") != cell["key"] or data.get("specification") != cell["specification"]:
                raise ValueError(f"existing keyed result is inconsistent: {target}")
            if data.get("status") != "complete" or "metrics" not in data:
                raise ValueError(f"existing result is incomplete: {target}; move it aside to rerun")
            reused.append(cell["key"])
        elif not dry_run and len(completed) < max_runs:
            c = cell["specification"]["config"]
            prepared, provenance = prepare_split(split, c["budget"], seed, max_train, max_test)
            kernel = make_statevector_kernel(c["budget"], c["reps"], c["entanglement"])
            result = {
                **cell,
                "status": "complete",
                "provenance": provenance,
                "metrics": evaluate_kernel(kernel, prepared),
            }
            write_new_json(target, result)
            completed.append(cell["key"])
    return {
        "manifest": str(manifest_path.resolve()),
        "dry_run": dry_run,
        "planned": len(cells),
        "completed": completed,
        "reused": reused,
        "remaining": len(cells) - len(completed) - len(reused),
    }


def overlap_circuits(
    X: np.ndarray, Y: np.ndarray | None = None, *, reps: int = 1, entanglement: str = "linear"
) -> tuple[list, list, tuple[int, int], bool]:
    """Measured U(y)^dagger U(x); P(all zeros) estimates squared fidelity.

    Symmetric mode executes the diagonal too: noisy self overlaps are not assumed
    to be 1. Maximum 32 rows per side, preventing unintended large remote jobs.
    """
    X = _matrix(X, "X")
    symmetric = Y is None
    Y = X if symmetric else _matrix(Y, "Y")
    if len(X) > 32 or len(Y) > 32 or X.shape[1] != Y.shape[1]:
        raise ValueError("overlap inputs need same width and <=32 samples per side")
    feature_map, _ = make_feature_map(X.shape[1], reps, entanglement)
    parameters = list(feature_map.parameters)
    states_x = [feature_map.assign_parameters(dict(zip(parameters, row))) for row in X]
    states_y = (
        states_x
        if symmetric
        else [feature_map.assign_parameters(dict(zip(parameters, row))) for row in Y]
    )
    circuits, pairs = [], []
    for i in range(len(X)):
        for j in range(i if symmetric else 0, len(Y)):
            circuit = states_x[i].copy()
            circuit.barrier()
            circuit.compose(states_y[j].inverse(), inplace=True)
            circuit.measure_all()
            circuits.append(circuit)
            pairs.append((i, j))
    return circuits, pairs, (len(X), len(Y)), symmetric


def aggregate_sampler_results(results, pairs, shape, symmetric: bool) -> tuple[np.ndarray, list]:
    """Assemble raw empirical zero-count frequencies, without PSD repair/mitigation."""
    results = list(results)
    if len(results) != len(pairs):
        raise ValueError("Sampler result count does not match submitted overlap circuits")
    kernel = np.full(shape, np.nan)
    shots = []
    for result, (i, j) in zip(results, pairs):
        counts = result.data.meas.get_counts()
        total = sum(counts.values())
        if total <= 0 or any(
            not isinstance(v, (int, np.integer)) or v < 0 for v in counts.values()
        ):
            raise ValueError("Sampler counts must contain nonnegative integers and positive shots")
        if any(
            not str(key).replace(" ", "") or set(str(key).replace(" ", "")) - {"0", "1"}
            for key in counts
        ):
            raise ValueError("Sampler meas.get_counts() must return binary-string keys")
        zeros = sum(
            value for key, value in counts.items() if set(str(key).replace(" ", "")) == {"0"}
        )
        kernel[i, j] = zeros / total
        if symmetric:
            kernel[j, i] = kernel[i, j]
        shots.append(int(total))
    if not np.isfinite(kernel).all():
        raise ValueError("overlap pair list does not cover the kernel matrix")
    return kernel, shots


def sampler_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    sampler=None,
    shots: int = 1024,
    reps: int = 1,
    entanglement: str = "linear",
    transpiler=None,
    on_submitted=None,
) -> dict:
    """Submit through an explicitly supplied SamplerV2-compatible object.

    Used for local reference/mocked testing and by the separately gated IBM call.
    No sampler is constructed implicitly, and no simulation fallback is permitted.
    """
    if sampler is None:
        raise ValueError("an explicit SamplerV2-compatible sampler is required")
    if not 1 <= shots <= 100000:
        raise ValueError("shots must be 1..100000")
    circuits, pairs, shape, symmetric = overlap_circuits(X, Y, reps=reps, entanglement=entanglement)
    submitted = transpiler.run(circuits) if transpiler is not None else circuits
    job = sampler.run(submitted, shots=shots)
    job_id = job.job_id()
    try:
        if on_submitted is not None:
            on_submitted(job_id)
        kernel, actual_shots = aggregate_sampler_results(job.result(), pairs, shape, symmetric)
    except Exception as exc:
        raise RuntimeError(f"Submitted job {job_id}; retrieve it instead of resubmitting") from exc
    return {
        "kernel": kernel.tolist(),
        "pairs": pairs,
        "shots_requested": shots,
        "shots_observed": actual_shots,
        "job_id": job_id,
        "circuit_count": len(circuits),
        "symmetric": symmetric,
        "psd_projected": False,
        "diagonal_forced_to_one": False,
        "mitigation": "none",
        "seed_scope": "transpilation only on hardware",
    }


def ibm_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    *,
    execute: bool = False,
    token: str | None = None,
    backend: str | None = None,
    instance: str | None = None,
    channel: str = "ibm_quantum_platform",
    shots: int = 1024,
    seed: int = 42,
    reps: int = 1,
    entanglement: str = "linear",
    mitigation: str = "none",
    on_submitted=None,
) -> dict:
    """Credential-free dry run by default; token must be explicitly supplied to execute.

    Does not search saved accounts or read environment variables. SamplerV2 ZNE is
    not implemented/supported here; asking for it is an error, never a simulation.
    """
    if mitigation != "none":
        raise ValueError(
            "Only mitigation='none' is supported. SamplerV2 ZNE is not "
            "supported by this harness; no silent estimator/simulator fallback."
        )
    if not 1 <= shots <= 100000:
        raise ValueError("shots must be 1..100000")
    circuits, pairs, shape, symmetric = overlap_circuits(X, Y, reps=reps, entanglement=entanglement)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "implementation": IMPLEMENTATION_VERSION,
        "mode": "ibm-execute" if execute else "credential-free-dry-run",
        "backend": backend,
        "instance": instance,
        "channel": channel,
        "seed_transpiler": seed,
        "seed_sampler": None,
        "shots_requested": shots,
        "circuit_count": len(circuits),
        "pairs": pairs,
        "kernel_shape": shape,
        "symmetric": symmetric,
        "reps": reps,
        "entanglement": entanglement,
        "qubits": circuits[0].num_qubits,
        "mitigation": "none",
        "input_sha256": hashlib.sha256(
            np.asarray(X, dtype=float).tobytes()
            + (b"symmetric" if Y is None else np.asarray(Y, dtype=float).tobytes())
        ).hexdigest(),
        "packages": package_versions(),
        "network_used": False,
    }
    if not execute:
        return plan
    if not token or not backend or not instance:
        raise ValueError(
            "execute requires explicit token, backend, and instance; "
            "saved-account credential discovery is disabled"
        )
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise RuntimeError(
            "IBM execution requires optional qiskit-ibm-runtime. "
            "Install it explicitly, then supply token/backend/instance."
        ) from exc
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
    target = service.backend(backend)
    transpiler = generate_preset_pass_manager(
        backend=target, optimization_level=1, seed_transpiler=seed
    )
    sampler = SamplerV2(mode=target)
    result = sampler_kernel(
        X,
        Y,
        sampler=sampler,
        shots=shots,
        reps=reps,
        entanglement=entanglement,
        transpiler=transpiler,
        on_submitted=on_submitted,
    )
    return {
        **plan,
        **result,
        "network_used": True,
        "backend_version": str(getattr(target, "backend_version", "unknown")),
        "optimization_level": 1,
    }
