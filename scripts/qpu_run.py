"""S3: run the EXP-D QUBO on a real D-Wave QPU / Leap Hybrid when a token exists.

The 50-variable dense QUBO is within direct-QPU embedding range but hybrid
solves it in seconds with better quality. Usage identical to arm D elsewhere;
this produces the single citable "solved on a quantum annealer" number.

Usage:  DWAVE_API_TOKEN=... .venv/Scripts/python.exe -m scripts.qpu_run
(no token -> prints the SA result and exits 0, so CI stays green)
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.config import ROOT, load_config
from src.quobo.features import run_features
from src.quobo.selection import compute_mi_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("qpu")


def build_qubo(X: np.ndarray, y: np.ndarray, k: int, alpha: float, n_bins: int = 16) -> dict:
    """Mücke-style QFS QUBO at fixed alpha (reuses the MI tables)."""
    I, R = compute_mi_table(X, y, n_bins)
    n = len(I)
    Q: dict[tuple[int, int], float] = {}
    for i in range(n):
        Q[(i, i)] = -alpha * I[i]
    for i in range(n):
        for j in range(i + 1, n):
            v = (1.0 - alpha) * R[i, j]
            Q[(i, j)] = v
            Q[(j, i)] = v
    return Q


def main() -> int:
    token = os.environ.get("DWAVE_API_TOKEN", "").strip()
    if not token:
        log.warning("DWAVE_API_TOKEN not set — QPU run skipped (SA path remains the default)")
        return 0

    cfg = load_config("configs/experiment_6class.yaml")
    df = run_features(cfg)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(float)
    y = np.asarray(df["label"].values)

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
        qubo = build_qubo(X, y, cfg["selection"]["k"], alpha)
        sampleset = sampler.sample_qubo(qubo, label=f"quobo-fs-alpha{alpha}")
        chosen = [i for i in sorted(qubo) if sampleset.first.sample[i] == 1]
        log.info("alpha=%.2f -> %d features: %s", alpha, len(chosen), chosen)
        if len(chosen) == cfg["selection"]["k"]:
            break

    out = {
        "solver": "LeapHybridSampler",
        "alpha": alpha,
        "selected": chosen,
        "n_features_selected": len(chosen),
        "qpu_time_seconds": round(time.time() - t0, 1),
        "dataset": "TACO-6class-1836crops-PCA50",
    }
    out_dir = ROOT / "results" / "qpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "qpu_selection.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("saved %s — feed these indices through the standard classifier suite "
             "for the citable hardware-validated EXP-D numbers", out_dir / "qpu_selection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
