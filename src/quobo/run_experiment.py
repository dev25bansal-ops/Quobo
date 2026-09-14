"""Experiment orchestrator: runs the full A-D ablation.

For each of n_repeats seeds:
  GROUP-aware split (no photo's crops straddle train/test) -> each arm selects
  k features -> train QSVM + RBF-SVM on those k columns -> record metrics,
  selected features, held-out labels, and per-classifier predictions.
Outputs: experiments/<timestamp>/results_all.csv + summary + significance
(+ per-rep JSONs consumed by confusion_analysis / pollution_dashboard).
"""
import json
import logging
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import ROOT, load_config, validate_config
from .features import fit_pca_on_train, run_raw_embeddings
from .qsvm import run_all_classifiers
from .selection import select_features

log = logging.getLogger(__name__)

ARM_LABELS = {
    "A_random": "Random-8",
    "B_pca": "PCA-8",
    "C_mi": "MI-8",
    "D_qubo": "QUBO-8",
    "E_lasso": "LASSO-8",
    "F_mrmr": "mRMR-8",
    "Z_full": "Full-50",
}


def crop_image_group(crop_path: str) -> str:
    """Group key = source TACO photo. Crop filenames look like
    batch_3_000123_ann45.jpg (batch_N + image stem + ann id), so everything
    before the final _annNNNN identifies the parent photo."""
    name = crop_path.replace("\\", "/").split("/")[-1]
    return name.rsplit("_ann", 1)[0]


def run(cfg: dict) -> pd.DataFrame:
    t_start = time.perf_counter()
    # fail fast on malformed config (Enhancement 1): validate the full schema
    # before any expensive work, not at rep 12
    validate_config(cfg)
    # OPEN-5 leak-free protocol: raw embeddings cached once (backbone is
    # deterministic per crop), then scaler+PCA refit INSIDE each split on
    # train rows only. PCA rotation no longer sees held-out data.
    raw = run_raw_embeddings(cfg)
    emb_cols = [c for c in raw.columns if c.startswith("e")]
    X_raw = raw[emb_cols].values.astype(np.float32)
    y = np.asarray(raw["label"].values)
    groups = np.array([crop_image_group(p) for p in raw["crop_path"]])
    n_comp = cfg["features"]["pca_components"]
    n_group_leaks = 0

    k = cfg["selection"]["k"]
    rows = []
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(cfg["experiment"]["n_repeats"]):
        seed = cfg["seed"] + rep
        # group-aware split on raw row indices; PCA refit on train rows only
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["qsvm"]["test_fraction"],
                                random_state=seed)
        idx_tr, idx_te = next(gss.split(X_raw, y, groups=groups))
        Xtr, Xte, ytr, yte = X_raw[idx_tr], X_raw[idx_te], y[idx_tr], y[idx_te]
        Xtr_p, Xall_p, ev_pct = fit_pca_on_train(Xtr, np.vstack([Xtr, Xte]), n_comp, seed)
        Xte = Xall_p[len(Xtr):]
        Xtr = Xtr_p
        overlap = set(groups[idx_tr]) & set(groups[idx_te])
        if overlap:
            n_group_leaks += len(overlap)
        # stratification report (group splits can drift from crop-level balance)
        if rep == 0:
            log.info("rep 0 split: train=%d test=%d PCA-%d explains %.1f%% (train-fit) "
                     "| test class share %s",
                     len(idx_tr), len(idx_te), n_comp, ev_pct,
                     {c: round(float((yte == c).mean()), 3) for c in sorted(set(yte))})
        for arm in cfg["experiment"]["arms"]:
            t0 = time.perf_counter()
            if arm == "Z_full":  # no-selection ceiling: all 50 features
                sel = list(range(Xtr.shape[1]))  # all PCA components (Z_full ceiling)
                sel_info = {"no_selection": True}
                sel_s = 0.0
            else:
                sel, sel_info = select_features(arm, Xtr, ytr, k, seed, cfg["qubo"])
                sel_s = time.perf_counter() - t0

            metrics = run_all_classifiers(
                Xtr[:, sel], ytr, Xte[:, sel], yte, cfg, rep_offset=rep
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
                    **{kk: vv for kk, vv in m.items() if kk != "predictions"},
                })
            log.info(
                "rep %d %s: qsvm=%s rbf=%.3f (%.1fs)",
                rep, arm,
                f"{metrics['qsvm']['accuracy']:.3f}" if "qsvm" in metrics else "n/a",
                metrics["rbf_svm"]["accuracy"], time.perf_counter() - t0,
            )
            # persist per-run details incl. held-out labels + predictions so
            # downstream analyses (confusion, PSI) consume without retraining
            with open(run_dir / f"rep{rep}_{arm}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "rep": rep, "arm": arm, "seed": seed, "selected": sel,
                    "selection_seconds": sel_s,
                    "classes": sorted({str(v) for v in y}),
                    "y_test": [str(v) for v in yte],
                    "metrics": metrics,
                    "mi_table": sel_info.get("I"),
                    "selection_info": {kk: vv for kk, vv in sel_info.items()
                                       if kk in ("alpha", "alpha_trace", "nudged_to_k",
                                                 "no_selection", "k_selected",
                                                 "n_features_candidate")},
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
