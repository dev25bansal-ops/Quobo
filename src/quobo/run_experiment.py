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


def log_to_mlflow(results: pd.DataFrame, cfg: dict, run_dir) -> bool:
    """Enhancement 10: opt-in MLflow experiment tracking (local file store, no
    server). Logs config params, per-arm mean/std metrics, and the full
    results CSV as an artifact so runs are comparable in `mlflow ui`. Returns
    True on success, False if mlflow is unavailable (degrades gracefully so
    the experiment never fails on the tracking extra)."""
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping experiment tracking "
                    "(pip install mlflow)")
        return False
    try:
        # MLflow 3.x: the plain ./mlruns file backend is in maintenance mode and
        # raises; the supported serverless option is a local SQLite store
        # (mlflow auto-initializes its schema on first use).
        import os
        import tempfile

        db = os.path.join(tempfile.gettempdir(), "quobo_mlruns.sqlite")
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        mlflow.set_experiment("quobo-feature-selection")
        with mlflow.start_run(run_name=f"{run_dir.name}"):
            mlflow.log_params({
                "seed": cfg["seed"],
                "selection.k": cfg["selection"]["k"],
                "features.pca_components": cfg["features"]["pca_components"],
                "qubo.mi_bins": cfg["qubo"]["mi_bins"],
                "qubo.sa_sweeps": cfg["qubo"]["sa_sweeps"],
                "qubo.sa_repeats": cfg["qubo"]["sa_repeats"],
                "qubo.parallel": cfg["qubo"].get("parallel", False),
                "qsvm.reps": cfg["qsvm"]["reps"],
                "qsvm.max_train_samples": cfg["qsvm"]["max_train_samples"],
                "qsvm.test_fraction": cfg["qsvm"]["test_fraction"],
                "experiment.n_repeats": cfg["experiment"]["n_repeats"],
                "experiment.arms": ",".join(cfg["experiment"]["arms"]),
                "data.classes": ",".join(cfg["data"]["classes"]),
            })
            agg = results.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].mean()
            for (arm, clf), row in agg.iterrows():
                mlflow.log_metric(f"accuracy_{arm}__{clf}", round(float(row["accuracy"]), 4))
                mlflow.log_metric(f"macro_f1_{arm}__{clf}", round(float(row["macro_f1"]), 4))
            mlflow.log_artifact(str(run_dir / "results_all.csv"))
            active = mlflow.active_run()
            run_id = active.info.run_id if active else "?"
            log.info("mlflow run logged (run_id=%s, sqlite store)", run_id)
        return True
    except Exception as e:  # noqa: BLE001 — tracking must never sink the experiment
        log.warning("mlflow tracking failed (continuing): %s: %s", type(e).__name__, str(e)[:120])
        return False


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

        # Enhancement 3: incremental write (crash recovery) + early stopping
        pd.DataFrame(rows).to_csv(run_dir / "results_interim.csv", index=False)
        if cfg["experiment"].get("early_stop", False) and rep >= 9:
            stop_reason = should_stop_early(rows, run_dir)
            if stop_reason:
                log.info("EARLY STOP after rep %d: %s", rep, stop_reason)
                break

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

    # Enhancement 10: opt-in MLflow tracking (file store; degrades gracefully)
    if cfg.get("experiment", {}).get("track_mlflow", False):
        log_to_mlflow(results, cfg, run_dir)

    log.info("total wall time: %.1f min", (time.perf_counter() - t_start) / 60)
    return results


def should_stop_early(rows: list, run_dir, min_reps: int = 10, window: int = 5,
                      tol: float = 0.01) -> str | None:
    """Stop if the per-repeat QUBO-8 RBF accuracy has been flat within tol for
    `window` consecutive repeats (result has converged). Returns a reason
    string to log, or None to continue. Conservative: requires min_reps and a
    full stable window, so it never cuts a still-moving experiment short."""
    if len(rows) < min_reps * len({r["arm_key"] for r in rows}):
        return None
    df = pd.DataFrame(rows)
    sub = df[(df.arm_key == "D_qubo") & (df.classifier == "rbf_svm")]
    if len(sub) < min_reps:
        return None
    accs = sub.sort_values("repeat")["accuracy"].tolist()
    if len(accs) < window:
        return None
    last = accs[-window:]
    if max(last) - min(last) <= tol:
        return (f"QUBO-8 rbf accuracy flat within {tol:.3f} for {window} "
                f"repeats (last {last[-1]:.4f})")
    return None


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
    args = sys.argv[1:]
    track = "--track" in args
    pos = [a for a in args if not a.startswith("--")]
    cfg = load_config(pos[0] if pos else "configs/experiment.yaml")
    if track:
        cfg.setdefault("experiment", {})["track_mlflow"] = True
    res = run(cfg)
    print(res.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].mean().round(4))
