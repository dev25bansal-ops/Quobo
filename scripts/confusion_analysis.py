"""Per-class confusion analysis for the headline comparison.

Consumes the persisted per-repeat predictions from the latest experiment run
(experiments/<ts>/rep*_D_qubo.json) — no retraining, no duplicated split/selection
logic (predictions are recorded by run_experiment via run_all_classifiers).

Usage: .venv/Scripts/python.exe -m scripts.confusion_analysis
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # scripts/ dir: importable as scripts.*
from src.quobo.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("confusion")


def latest_run_dir() -> Path:
    """Locate the most recent run directory.

    Prefers a run whose manifest says ``complete`` (a ``running``/``failed``
    run's per-rep artifacts may be partial); falls back to the newest directory
    that has a manifest if none are complete, so the artifact-hash check in
    _verify_artifacts still catches tampering. If no manifests exist at all
    (legacy/bare dirs), returns the newest directory — the hash verification
    in _verify_artifacts then gates the actual consumption."""
    experiments = ROOT / "experiments"
    if not experiments.is_dir() or not any(experiments.iterdir()):
        raise SystemExit("no experiment runs found — run src.quobo.run_experiment first")
    complete = []
    with_manifest = []
    for d in sorted(experiments.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        with_manifest.append(d)
        try:
            status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            status = None
        if status == "complete":
            complete.append(d)
    if complete:
        return complete[-1]
    if with_manifest:
        log.warning(
            "no completed run found; using %s (status may be incomplete — "
            "per-rep artifacts are hash-verified but the run was not finalized)",
            with_manifest[-1].name,
        )
        return with_manifest[-1]
    # No manifests at all (legacy/bare dirs) — newest dir, hash check gates it
    return sorted(experiments.iterdir())[-1]


def _verify_artifacts(run_dir: Path, reps: list[Path]) -> None:
    """Sec-note: confirm every rep file matches the sha256 recorded in the run
    manifest before trusting its predictions. Mirrors pollution_dashboard's
    load_predictions check; a truncated or substituted rep is rejected."""
    import hashlib

    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"run {run_dir.name} has no run_manifest.json — cannot verify")
    artifacts = json.loads(manifest_path.read_text(encoding="utf-8")).get("artifacts") or {}

    def sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    for rep in reps:
        recorded = artifacts.get(rep.name)
        if not recorded or sha256_of(rep) != recorded.get("sha256"):
            raise SystemExit(f"{rep.name} does not match the run manifest (corrupt or substituted)")


def main() -> None:
    run_dir = latest_run_dir()
    reps = sorted(run_dir.glob("rep*_D_qubo.json"))
    if not reps:
        raise SystemExit(
            f"run {run_dir.name} has no rep*_D_qubo.json files — confusion "
            "analysis currently targets the QUBO arm; run with D_qubo in arms"
        )
    log.info("using run %s (%d repeats)", run_dir.name, len(reps))
    _verify_artifacts(run_dir, reps)

    with open(reps[0], encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    agg = {
        "qsvm": np.zeros((len(classes), len(classes)), dtype=int),
        "rbf_svm": np.zeros((len(classes), len(classes)), dtype=int),
    }
    per_class_rows = []

    for rep_file in reps:
        with open(rep_file, encoding="utf-8") as f:
            j = json.load(f)
        yte = np.array(j["y_test"])

        # Only aggregate/classify classifiers that actually exist in this rep
        # (QSVM is skipped when k > max_qsvm_features).
        for clf in ("rbf_svm", "qsvm"):
            if clf not in j["metrics"]:
                continue
            pred = np.array(j["metrics"][clf]["predictions"])
            agg[clf] += confusion_matrix(yte, pred, labels=classes)

        p, r, f1, s = precision_recall_fscore_support(
            yte, np.array(j["metrics"]["rbf_svm"]["predictions"]), labels=classes, zero_division=0
        )
        for c, pi, ri, fi, si in zip(classes, p, r, f1, s, strict=True):
            per_class_rows.append(
                {
                    "repeat": j.get("rep", rep_file.stem),
                    "classifier": "rbf_svm",
                    "class": c,
                    "precision": pi,
                    "recall": ri,
                    "f1": fi,
                    "support": int(si),
                }
            )
        # QSVM per-class rows too (audit: previously RBF-only) — but only if
        # QSVM actually ran for this rep (it is skipped when k > max_qsvm_features)
        if "qsvm" in j["metrics"]:
            p, r, f1, s = precision_recall_fscore_support(
                yte, np.array(j["metrics"]["qsvm"]["predictions"]), labels=classes, zero_division=0
            )
            for c, pi, ri, fi, si in zip(classes, p, r, f1, s, strict=True):
                per_class_rows.append(
                    {
                        "repeat": j.get("rep", rep_file.stem),
                        "classifier": "qsvm",
                        "class": c,
                        "precision": pi,
                        "recall": ri,
                        "f1": fi,
                        "support": int(si),
                    }
                )

    out = ROOT / "results" / "confusion"
    out.mkdir(parents=True, exist_ok=True)

    pc = pd.DataFrame(per_class_rows)
    pc_mean = (
        pc.groupby(["class", "classifier"])[["precision", "recall", "f1", "support"]]
        .mean()
        .round(3)
    )
    pc_mean.to_csv(out / "per_class_metrics.csv")

    for clf, cm in agg.items():
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j,
                    i,
                    f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                )
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(
            f"{'QSVM' if clf == 'qsvm' else 'RBF-SVM'} — QUBO-8, aggregated over {len(reps)} repeats"
        )
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{clf}.png", dpi=150)
        plt.close(fig)
        np.savetxt(
            out / f"confusion_{clf}.csv",
            cm,
            delimiter=",",
            fmt="%d",
            header=",".join(classes),
            comments="",
        )

    print(pc_mean.to_string())
    print("saved to", out)


if __name__ == "__main__":
    main()
