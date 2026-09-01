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

    # paired significance tests: every arm vs every arm, per classifier,
    # on per-repeat accuracy. Wilcoxon signed-rank (exact via normal approx
    # at n>=25) + paired t, Holm-corrected across the family of comparisons.
    sig_rows = paired_significance(results)
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(run_dir / "significance.csv", index=False)
        pd.DataFrame(sig_rows).to_csv(
            ROOT / "results" / "tables" / f"significance_{stamp}.csv", index=False
        )
        log.info("significance: %d comparisons written", len(sig_rows))

    log.info("total wall time: %.1f min", (time.perf_counter() - t_start) / 60)
    return results


def paired_significance(results: pd.DataFrame) -> list[dict]:
    """Holm-corrected Wilcoxon signed-rank + paired t tests on per-repeat accuracy."""
    from scipy.stats import ttest_rel, wilcoxon

    def holm(pvals: list[float]) -> list[float]:
        """Holm-Bonferroni step-down adjusted p-values."""
        m = len(pvals)
        order = sorted(range(m), key=lambda i: pvals[i])
        adj, running = [0.0] * m, 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * pvals[i])
            adj[i] = min(1.0, running)
        return adj

    arms = sorted(results["arm"].unique())
    comparisons = []
    for clf in sorted(results["classifier"].unique()):
        sub = results[results["classifier"] == clf]
        wide = sub.pivot_table(index="repeat", columns="arm", values="accuracy")
        for a, b in [(x, y) for i, x in enumerate(arms) for y in arms[i + 1 :]]:
            if a not in wide or b not in wide:
                continue
            diff = (wide[a] - wide[b]).dropna()
            comparisons.append({"clf": clf, "arm_a": a, "arm_b": b, "diff": diff})

    if not comparisons:
        return []

    raw_p = [
        float(wilcoxon(c["diff"]).pvalue) if c["diff"].abs().sum() > 0 else 1.0
        for c in comparisons
    ]
    p_corr = holm(raw_p)
    out = []
    for c, p in zip(comparisons, p_corr):
        d = c["diff"]
        _, tp = ttest_rel(d, np.zeros_like(d))
        out.append({
            "classifier": c["clf"],
            "arm_a": c["arm_a"], "arm_b": c["arm_b"],
            "mean_diff": round(float(d.mean()), 4),
            "wilcoxon_p_holm": round(float(p), 4),
            "paired_t_p_holm": round(float(tp), 4),
            "significant_0.05": bool(p < 0.05),
        })
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml")
    res = run(cfg)
    print(res.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].mean().round(4))
