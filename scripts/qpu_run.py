"""S3: run the EXP-D QUBO on a real D-Wave QPU / Leap Hybrid when a token exists.

The 50-variable dense QUBO is within direct-QPU embedding range but hybrid
solves it in seconds with better quality. Usage identical to arm D elsewhere;
this produces the single citable "solved on a quantum annealer" number.

L-06 (M-01): this script used to call the LEGACY ``run_features`` path, whose
PCA rotation is fitted on ALL rows. That is exactly the transductive leak the
paper's protocol forbids (OPEN-5), and it made the hardware number
incomparable to the leak-free SA baseline it is supposed to validate. The
script now follows the same protocol as the main pipeline: raw embeddings ->
group-aware split -> scaler+PCA refit on TRAIN ROWS ONLY -> MI table and QUBO
built from the train-PCA features of that same split.

L-05 (M-01): the result JSON is published with ``publish_exclusive`` (temp
file -> fsync -> exclusive link), so a crash can never leave a truncated
``qpu_selection.json`` that later reads as valid, and a valid prior artifact
is never silently overwritten.

Usage:  DWAVE_API_TOKEN=... .venv/Scripts/python.exe -m scripts.qpu_run
(no token -> prints the SA result and exits 0, so CI stays green)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.atomic import publish_exclusive, publish_replace  # noqa: E402
from src.quobo.config import ROOT, load_config  # noqa: E402
from src.quobo.features import fit_pca_on_train, run_raw_embeddings  # noqa: E402
from src.quobo.selection import compute_mi_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("qpu")


@dataclass(frozen=True)
class LeakFreeSplit:
    """One group-aware split with a TRAIN-ONLY PCA rotation (OPEN-5 / L-06).

    ``X_train_pca`` / ``X_test_pca`` are produced by ``fit_pca_on_train``:
    the scaler and the PCA are fitted on the train rows and merely applied to
    the test rows, so no held-out sample influences the rotation.
    """

    X_train_pca: np.ndarray
    y_train: np.ndarray
    X_test_pca: np.ndarray
    y_test: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray
    explained_variance_pct: float
    n_overlap_groups: int


def make_leak_free_split(
    cfg: dict, raw: pd.DataFrame | None = None, seed: int | None = None
) -> LeakFreeSplit:
    """Group-aware split + train-only PCA fit, mirroring the main pipeline.

    The grouping key is ``crop_image_group`` (the source TACO photo), so no
    photo's crops straddle train and test — the same invariant the experiment
    runner enforces (LKG-002). The split uses ``cfg['seed']`` (rep 0 of the
    experiment) so the hardware-selected feature set is comparable with the
    SA arm D result for that repeat.
    """
    if raw is None:
        raw = run_raw_embeddings(cfg)
    # Imported here, not at module scope: run_experiment pulls in the QVM
    # stack (qiskit), which the QPU script never uses. Deferring it keeps the
    # no-token CI path (which returns before any split) cheap. The grouping
    # helper itself must be the SAME helper the pipeline uses, or the two
    # splits would differ and the numbers would not be comparable.
    from src.quobo.run_experiment import crop_image_group

    emb_cols = [c for c in raw.columns if c.startswith("e")]
    if not emb_cols:
        raise ValueError("raw embeddings frame has no 'e*' embedding columns")
    X_raw = raw[emb_cols].values.astype(np.float32)
    y = np.asarray(raw["label"].values)
    groups = np.array([crop_image_group(p) for p in raw["crop_path"]])

    seed = int(cfg["seed"]) if seed is None else int(seed)
    n_comp = int(cfg["features"]["pca_components"])
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["qsvm"]["test_fraction"], random_state=seed)
    idx_tr, idx_te = next(gss.split(X_raw, y, groups=groups))
    Xtr, Xte = X_raw[idx_tr], X_raw[idx_te]
    # scaler+PCA fitted on the train rows only; the test rows are transformed
    # with that fitted state (never with a fit of their own).
    Xtr_p, Xall_p, ev_pct = fit_pca_on_train(Xtr, np.vstack([Xtr, Xte]), n_comp, seed)
    Xte_p = Xall_p[len(Xtr) :]

    overlap = set(groups[idx_tr]) & set(groups[idx_te])
    if overlap:
        # GroupShuffleSplit makes this structurally impossible; a non-empty
        # intersection means the split itself is broken, and the QUBO built on
        # these rows would no longer be comparable with the SA baseline.
        raise RuntimeError(
            f"{len(overlap)} source photos straddle train/test — the leak-free "
            "invariant is violated, refusing to build a QUBO from these rows"
        )
    return LeakFreeSplit(
        X_train_pca=Xtr_p,
        y_train=y[idx_tr],
        X_test_pca=Xte_p,
        y_test=y[idx_te],
        train_groups=groups[idx_tr],
        test_groups=groups[idx_te],
        explained_variance_pct=ev_pct,
        n_overlap_groups=len(overlap),
    )


def build_qubo(X: np.ndarray, y: np.ndarray, alpha: float, n_bins: int = 16) -> dict:
    """Mücke-style QFS QUBO at fixed alpha (reuses the MI tables).

    NOTE: the QUBO does not constrain cardinality. The exactly-k property
    comes from the alpha search in main() (as in src/quobo/selection.py), not
    from this matrix — so the budget `k` is deliberately NOT a parameter here.

    C-02: only the DIAGONAL and the UPPER TRIANGLE are emitted. dimod's
    ``from_qubo`` SUMS the two orientations of a pair, so emitting both
    ``(i, j)`` and ``(j, i)`` doubles every redundancy term — the QPU would
    then optimize ``2*(1-alpha)*R_ij`` instead of the ``(1-alpha)*R_ij`` the
    module docstring, the config and the paper document (Mücke et al.,
    arXiv:2203.13261). ``src/quobo/selection.py`` already emits the upper
    triangle only; this script must match it bit for bit, otherwise the
    hardware result is not comparable with the SA baseline.
    """
    I, R = compute_mi_table(X, y, n_bins)  # noqa: E741
    n = len(I)
    Q: dict[tuple[int, int], float] = {}
    for i in range(n):
        Q[(i, i)] = -alpha * I[i]
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = (1.0 - alpha) * R[i, j]
    return Q


def _publish_result(dest: Path, payload: dict) -> None:
    """Publish the result JSON atomically, never clobbering a valid artifact.

    ``publish_exclusive`` is the right primitive for a *result*: a second
    writer must not silently replace a good artifact with its own (the
    published JSON is what the paper's numbers are read from). If the
    existing file is not readable as JSON it is a crash remnant from the old
    non-atomic write path — that one is replaced, because it carries no
    information a consumer could trust.
    """
    data = json.dumps(payload, indent=2).encode("utf-8")
    try:
        publish_exclusive(dest, data)
        return
    except FileExistsError:
        pass
    try:
        json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        publish_replace(dest, data)
        log.warning("replaced an unreadable prior artifact at %s", dest)
        return
    raise FileExistsError(
        f"{dest} already exists and is a valid JSON artifact — refusing to "
        "overwrite it (move it aside if the new run should replace it)"
    )


def main() -> int:
    token = os.environ.get("DWAVE_API_TOKEN", "").strip()
    if not token:
        log.warning("DWAVE_API_TOKEN not set — QPU run skipped (SA path remains the default)")
        return 0

    cfg = load_config("configs/experiment_6class.yaml")
    # L-06: leak-free protocol — group-aware split, PCA refit on train rows
    # only, and the MI table + QUBO built from those same train rows.
    split = make_leak_free_split(cfg)
    X = split.X_train_pca
    y = split.y_train
    log.info(
        "leak-free split: train=%d test=%d PCA-%d (train-fit, %.1f%% variance), group overlap=%d",
        len(X),
        len(split.X_test_pca),
        X.shape[1],
        split.explained_variance_pct,
        split.n_overlap_groups,
    )

    from dwave.cloud import Client
    from dwave.system import LeapHybridSampler

    with Client.from_config(token=token) as client:
        solvers = [s.id for s in client.get_solvers()]
    log.info("Leap solvers visible: %s", solvers)

    t0 = time.time()
    sampler = LeapHybridSampler(token=token)
    # alpha chosen by the SA path's own bisection (persisted in rep JSONs);
    # 0.5 is the canonical midpoint documented in selection.py
    for alpha in (0.5, 0.6, 0.7):
        # mi_bins comes from the config so the MI table (and hence the QUBO)
        # is built exactly like the SA arm it is compared against.
        qubo = build_qubo(X, y, alpha, n_bins=cfg["qubo"]["mi_bins"])
        sampleset = sampler.sample_qubo(qubo, label=f"quobo-fs-alpha{alpha}")
        chosen = [i for i in range(X.shape[1]) if sampleset.first.sample[i] == 1]
        log.info("alpha=%.2f -> %d features: %s", alpha, len(chosen), chosen)
        if len(chosen) == cfg["selection"]["k"]:
            break

    out = {
        "solver": "LeapHybridSampler",
        "alpha": alpha,
        "selected": chosen,
        "n_features_selected": len(chosen),
        "requested_k": cfg["selection"]["k"],
        "cardinality_ok": len(chosen) == cfg["selection"]["k"],
        "qpu_time_seconds": round(time.time() - t0, 1),
        "dataset": "TACO-6class-1836crops-PCA50",
        # provenance of the leak-free protocol (OPEN-5): the rotation was
        # fitted on the train rows of this group-aware split only.
        "protocol": "leak_free_group_split_train_only_pca",
        "pca_fit_on": "train_rows_only",
        "n_train_rows": int(len(X)),
        "n_test_rows": int(len(split.X_test_pca)),
        "group_overlap_events": int(split.n_overlap_groups),
        "seed": int(cfg["seed"]),
        "pca_components": int(X.shape[1]),
    }
    if not out["cardinality_ok"]:
        log.warning(
            "QPU sample has %d features but k=%d was requested; result is not "
            "directly comparable to exactly-k experiment arms",
            len(chosen),
            cfg["selection"]["k"],
        )
    out_dir = ROOT / "results" / "qpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "qpu_selection.json"
    try:
        _publish_result(dest, out)
    except FileExistsError as exc:
        log.error("%s", exc)
        return 1
    log.info(
        "saved %s — feed these indices through the standard classifier suite "
        "for the citable hardware-validated EXP-D numbers",
        dest,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
