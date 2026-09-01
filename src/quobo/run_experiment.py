"""Experiment orchestrator: runs the full A-D ablation.

For each of n_repeats seeds:
  stratified split -> each arm selects k features -> train QSVM + RBF-SVM on
  those k columns -> record metrics + selected features.
Outputs: experiments/<timestamp>/results.json + results/tables/summary.csv
"""
import json
import logging
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ROOT, load_config
from .features import run_features
from .qsvm import run_all_classifiers
from .selection import select_features

log = logging.getLogger(__name__)

ARM_LABELS = {
    "A_random": "Random-8",
    "B_pca": "PCA-8",
    "C_mi": "MI-8",
    "D_qubo": "QUBO-8",
}


def run(cfg: dict) -> pd.DataFrame:
    t_start = time.perf_counter()
    df = run_features(cfg)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    y = df["label"].values

    k = cfg["selection"]["k"]
    rows = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(cfg["experiment"]["n_repeats"]):
        seed = cfg["seed"] + rep
        Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
            X, y, np.arange(len(y)),
            test_size=cfg["qsvm"]["test_fraction"],
            random_state=seed, stratify=y,
        )
        for arm in cfg["experiment"]["arms"]:
            t0 = time.perf_counter()
            sel, sel_info = select_features(arm, Xtr, ytr, k, seed, cfg["qubo"])
            sel_s = time.perf_counter() - t0

            metrics = run_all_classifiers(
                Xtr[:, sel], ytr, Xte[:, sel], yte, cfg
            )
            for clf, m in metrics.items():
                rows.append({
                    "repeat": rep,
                    "seed": seed,
                    "arm": ARM_LABELS[arm],
                    "arm_key": arm,
                    "classifier": clf,
                    "selected": ",".join(map(str, sel)),
                    "selection_seconds": sel_s,
                    **m,
                })
            log.info(
                "rep %d %s: qsvm=%.3f rbf=%.3f (%.1fs)",
                rep, arm, metrics["qsvm"]["accuracy"],
                metrics["rbf_svm"]["accuracy"], time.perf_counter() - t0,
            )
            # persist per-run details
            with open(run_dir / f"rep{rep}_{arm}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "arm": arm, "seed": seed, "selected": sel,
                    "selection_seconds": sel_s,
                    "metrics": metrics,
                    "mi_table": sel_info.get("I"),
                }, f, indent=1)

    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "results_all.csv", index=False)

    summary = (
        results.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(ROOT / "results" / "tables" / f"summary_{stamp}.csv")
    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    log.info("total wall time: %.1f min", (time.perf_counter() - t_start) / 60)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml")
    res = run(cfg)
    print(res.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].mean().round(4))
