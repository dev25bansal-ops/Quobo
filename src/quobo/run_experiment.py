"""Experiment orchestrator: runs the full A-D ablation.

For each of n_repeats seeds, a group-aware split is made (no photo's crops
straddle train/test), then each arm selects k features, trains QSVM + RBF-SVM
on those k columns, and records metrics, selected features, held-out labels,
and per-classifier predictions.

Outputs (under ``experiments/<timestamp>/``): ``results_all.csv``, the summary
CSV + significance CSV under ``results/tables/``, and per-rep JSONs consumed by
``confusion_analysis`` / ``pollution_dashboard``.
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

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
    # Z_full is the no-selection ceiling over the *actual* PCA pool; its label
    # is derived from features.pca_components at use time (see arm_label).
}


def arm_label(arm: str, n_comp: int) -> str:
    """Display label for an arm. The Z_full ceiling is labelled ``Full-{n_comp}``
    so it tracks the configured PCA pool instead of a hardcoded 50 (Q17/Q23)."""
    if arm == "Z_full":
        return f"Full-{n_comp}"
    return ARM_LABELS[arm]


MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
MANIFEST_IMPLEMENTATION = "run-manifest-v1"
REP_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a same-directory temp file + ``os.replace`` so a
    consumer never observes a half-written manifest or per-rep artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _artifact_inventory(run_dir: Path) -> dict:
    """Hash the run's persisted artifacts so consumers can detect truncation or
    substitution (Q04: reject incomplete or misaligned runs)."""
    names = {p.name for p in run_dir.glob("rep*_*.json")}
    for name in ("results_all.csv", "results_interim.csv", "config_used.json", "significance.csv"):
        if (run_dir / name).exists():
            names.add(name)
    return {
        name: {"sha256": _sha256_file(run_dir / name), "bytes": (run_dir / name).stat().st_size}
        for name in sorted(names)
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
        log.warning("mlflow not installed — skipping experiment tracking " "(pip install mlflow)")
        return False
    try:
        # MLflow 3.x: the plain ./mlruns file backend is in maintenance mode and
        # raises; the supported serverless option is a local SQLite store
        # (mlflow auto-initializes its schema on first use). Q24: a durable
        # tracking location can be pinned with QUOBO_MLFLOW_TRACKING_URI; the
        # ephemeral temp SQLite store is only the fallback.
        import os
        import tempfile

        env_uri = os.environ.get("QUOBO_MLFLOW_TRACKING_URI")
        if env_uri:
            tracking_uri = env_uri
            log.info("mlflow tracking store from QUOBO_MLFLOW_TRACKING_URI")
        else:
            db = os.path.join(tempfile.gettempdir(), "quobo_mlruns.sqlite")
            tracking_uri = f"sqlite:///{db}"
            log.info("mlflow tracking store: default ephemeral sqlite temp store")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("quobo-feature-selection")
        with mlflow.start_run(run_name=f"{run_dir.name}"):
            mlflow.log_params(
                {
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
                }
            )
            agg = results.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].mean()
            for (arm, clf), row in agg.iterrows():
                mlflow.log_metric(f"accuracy_{arm}__{clf}", round(float(row["accuracy"]), 4))
                mlflow.log_metric(f"macro_f1_{arm}__{clf}", round(float(row["macro_f1"]), 4))
            mlflow.log_artifact(str(run_dir / "results_all.csv"))
            active = mlflow.active_run()
            run_id = active.info.run_id if active else "?"
            log.info("mlflow run logged (run_id=%s)", run_id)
        return True
    except Exception as e:  # noqa: BLE001 — tracking must never sink the experiment
        log.warning("mlflow tracking failed (continuing): %s: %s", type(e).__name__, str(e)[:120])
        return False


def run(cfg: dict, run_id: str | None = None, config_path: str | None = None) -> pd.DataFrame:
    """Validate the config, open a run manifest in the ``running`` state, execute
    the ablation, and mark the run ``complete`` only after every artifact is
    written. Any failure marks the manifest ``failed`` and re-raises, so
    consumers can refuse partial runs instead of guessing (Q04)."""
    # fail fast on malformed config (Enhancement 1): validate the full schema
    # before any expensive work, not at rep 12. Q17: the rest of the run uses
    # the validated model's normalized dump, so schema defaults (parallel,
    # early_stop, max_qsvm_features, ...) are actually in effect.
    model = validate_config(cfg)
    cfg = model.model_dump()
    log.info("config validated against schema; using normalized values")
    stamp = run_id or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "implementation": MANIFEST_IMPLEMENTATION,
        "run_id": stamp,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "finished_at": None,
        "config_sha256": _config_digest(cfg),
        "config_path": config_path,
        "seed": cfg["seed"],
        "classes": list(cfg["data"]["classes"]),
        "arms": list(cfg["experiment"]["arms"]),
        "n_repeats_requested": cfg["experiment"]["n_repeats"],
        "reps_completed": 0,
        "artifacts": {},
        "note": (
            "Consumers must require status=='complete' and verify the "
            "artifact hashes before trusting predictions."
        ),
    }
    _write_json_atomic(run_dir / MANIFEST_NAME, manifest)
    try:
        results = _run_body(cfg, run_dir, stamp, manifest)
    except Exception as exc:  # record the failure, then re-raise
        manifest.update(
            {
                "status": "failed",
                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": _artifact_inventory(run_dir),
            }
        )
        _write_json_atomic(run_dir / MANIFEST_NAME, manifest)
        raise
    manifest.update(
        {
            "status": "complete",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "artifacts": _artifact_inventory(run_dir),
        }
    )
    _write_json_atomic(run_dir / MANIFEST_NAME, manifest)
    return results


def _run_body(cfg: dict, run_dir: Path, stamp: str, manifest: dict) -> pd.DataFrame:
    t_start = time.perf_counter()
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
    n_reps_done = 0

    for rep in range(cfg["experiment"]["n_repeats"]):
        seed = cfg["seed"] + rep
        # group-aware split on raw row indices; PCA refit on train rows only
        gss = GroupShuffleSplit(
            n_splits=1, test_size=cfg["qsvm"]["test_fraction"], random_state=seed
        )
        idx_tr, idx_te = next(gss.split(X_raw, y, groups=groups))
        Xtr, Xte, ytr, yte = X_raw[idx_tr], X_raw[idx_te], y[idx_tr], y[idx_te]
        Xtr_p, Xall_p, ev_pct = fit_pca_on_train(Xtr, np.vstack([Xtr, Xte]), n_comp, seed)
        Xte = Xall_p[len(Xtr) :]
        Xtr = Xtr_p
        overlap = set(groups[idx_tr]) & set(groups[idx_te])
        if overlap:
            n_group_leaks += len(overlap)
        # stratification report (group splits can drift from crop-level balance)
        if rep == 0:
            log.info(
                "rep 0 split: train=%d test=%d PCA-%d explains %.1f%% (train-fit) "
                "| test class share %s",
                len(idx_tr),
                len(idx_te),
                n_comp,
                ev_pct,
                {c: round(float((yte == c).mean()), 3) for c in sorted(set(yte))},
            )
        for arm in cfg["experiment"]["arms"]:
            t0 = time.perf_counter()
            label = arm_label(arm, n_comp)
            if arm == "Z_full":  # no-selection ceiling: all PCA components
                sel = list(range(Xtr.shape[1]))  # all PCA components (Z_full ceiling)
                sel_info = {"no_selection": True}
                sel_s = 0.0
            else:
                sel, sel_info = select_features(arm, Xtr, ytr, k, seed, cfg["qubo"])
                sel_s = time.perf_counter() - t0

            metrics = run_all_classifiers(Xtr[:, sel], ytr, Xte[:, sel], yte, cfg, rep_offset=rep)
            for clf, m in metrics.items():
                rows.append(
                    {
                        "repeat": rep,
                        "seed": seed,
                        "arm": label,
                        "arm_key": arm,
                        "classifier": clf,
                        "selected": ",".join(map(str, sel)),
                        "selection_seconds": sel_s,
                        **{kk: vv for kk, vv in m.items() if kk != "predictions"},
                    }
                )
            log.info(
                "rep %d %s: qsvm=%s rbf=%.3f (%.1fs)",
                rep,
                arm,
                f"{metrics['qsvm']['accuracy']:.3f}" if "qsvm" in metrics else "n/a",
                metrics["rbf_svm"]["accuracy"],
                time.perf_counter() - t0,
            )
            # persist per-run details incl. held-out labels + predictions so
            # downstream analyses (confusion, PSI) consume without retraining.
            # test_ids/test_groups are the exact held-out crop identity, aligned
            # 1:1 with y_test and each classifier's predictions (Q04 provenance);
            # consumers must never re-derive the split from the current config.
            _write_json_atomic(
                run_dir / f"rep{rep}_{arm}.json",
                {
                    "schema_version": REP_SCHEMA_VERSION,
                    "run_id": stamp,
                    "rep": rep,
                    "arm": arm,
                    "arm_label": label,
                    "seed": seed,
                    "selected": sel,
                    "selection_seconds": sel_s,
                    "classes": sorted({str(v) for v in y}),
                    "y_test": [str(v) for v in yte],
                    "test_ids": [str(p) for p in raw["crop_path"].values[idx_te]],
                    "test_groups": [str(g) for g in groups[idx_te]],
                    "metrics": metrics,
                    "mi_table": sel_info.get("I"),
                    "selection_info": {
                        kk: vv
                        for kk, vv in sel_info.items()
                        if kk
                        in (
                            "alpha",
                            "alpha_trace",
                            "nudged_to_k",
                            "no_selection",
                            "k_selected",
                            "n_features_candidate",
                        )
                    },
                },
            )
            n_reps_done = rep + 1

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
    # at n>=25) and paired t are each Holm-corrected as separate families.
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

    manifest["reps_completed"] = n_reps_done
    log.info("total wall time: %.1f min", (time.perf_counter() - t_start) / 60)
    return results


def should_stop_early(
    rows: list, run_dir, min_reps: int = 10, window: int = 5, tol: float = 0.01
) -> str | None:
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
        return (
            f"QUBO-8 rbf accuracy flat within {tol:.3f} for {window} "
            f"repeats (last {last[-1]:.4f})"
        )
    return None


def paired_significance(results: pd.DataFrame) -> list[dict]:
    """Adjust Wilcoxon and paired-t families separately across all comparisons.

    Insufficient paired observations or all-zero differences receive p=1.
    The significance flag uses the unrounded adjusted Wilcoxon p-value.
    """
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
        float(wilcoxon(c["diff"]).pvalue) if c["diff"].abs().sum() > 0 else 1.0 for c in comparisons
    ]
    p_corr = holm(raw_p)
    raw_t_p = []
    for c in comparisons:
        d = c["diff"]
        if len(d) < 2 or not d.ne(0).any():
            raw_t_p.append(1.0)
        else:
            _, tp = ttest_rel(d, np.zeros_like(d))
            raw_t_p.append(float(tp) if np.isfinite(tp) else 1.0)
    t_corr = holm(raw_t_p)
    out = []
    for c, p, tp in zip(comparisons, p_corr, t_corr):
        d = c["diff"]
        out.append(
            {
                "classifier": c["clf"],
                "arm_a": c["arm_a"],
                "arm_b": c["arm_b"],
                "mean_diff": round(float(d.mean()), 4),
                "wilcoxon_p_holm": round(float(p), 4),
                "paired_t_p_holm": round(float(tp), 4),
                "significant_0.05": bool(p < 0.05),
            }
        )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the feature-selection experiment")
    parser.add_argument("config", nargs="?", default="configs/experiment_6class.yaml")
    parser.add_argument("--track", action="store_true", help="Log results to local MLflow")
    parser.add_argument(
        "--run-id", default=None, help="Explicit run directory name (default: timestamp)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config(args.config)
    if args.track:
        cfg.setdefault("experiment", {})["track_mlflow"] = True
    res = run(cfg, run_id=args.run_id, config_path=args.config)
    print(res.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].mean().round(4))
