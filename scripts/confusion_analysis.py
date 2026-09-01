"""Per-class confusion analysis for the headline comparison.

Retrains the QUBO-8 arm across the same 5 seeds as the main experiment,
collects held-out confusion matrices for QSVM + RBF-SVM, and reports
per-class precision/recall plus aggregated confusion heatmaps.

Usage: .venv/Scripts/python.exe -m scripts.confusion_analysis
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.config import ROOT, load_config
from src.quobo.features import run_features
from src.quobo.qsvm import make_qsvc, run_all_classifiers, scale_for_feature_map
from src.quobo.selection import select_features
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("confusion")


def main() -> None:
    cfg = load_config("configs/experiment_6class.yaml")
    df = run_features(cfg)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X, y = df[feat_cols].values, df["label"].values
    classes = sorted(np.unique(y))
    k = cfg["selection"]["k"]
    n_rep = cfg["experiment"]["n_repeats"]

    agg = {"qsvm": np.zeros((len(classes), len(classes)), dtype=int),
           "rbf_svm": np.zeros((len(classes), len(classes)), dtype=int)}
    per_class_rows = []

    for rep in range(n_rep):
        seed = cfg["seed"] + rep
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=cfg["qsvm"]["test_fraction"],
            random_state=seed, stratify=y)
        sel, _ = select_features("D_qubo", Xtr, ytr, k, seed, cfg["qubo"])

        max_n = cfg["qsvm"]["max_train_samples"]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ytr), size=min(max_n, len(ytr)), replace=False)

        qsvc, _ = make_qsvc(k, cfg)
        qsvc.class_weight = "balanced"
        qsvc.fit(scale_for_feature_map(Xtr[np.ix_(idx, sel)]), np.asarray(ytr)[idx])
        pred_q = qsvc.predict(scale_for_feature_map(Xte[:, sel]))
        agg["qsvm"] += confusion_matrix(yte, pred_q, labels=classes)

        rbf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
        rbf.fit(Xtr[np.ix_(idx, sel)], np.asarray(ytr)[idx])
        pred_r = rbf.predict(Xte[:, sel])
        agg["rbf_svm"] += confusion_matrix(yte, pred_r, labels=classes)

        p, r, f, s = precision_recall_fscore_support(yte, pred_r, labels=classes, zero_division=0)
        for c, pi, ri, fi, si in zip(classes, p, r, f, s):
            per_class_rows.append({"repeat": rep, "classifier": "rbf_svm", "class": c,
                                   "precision": pi, "recall": ri, "f1": fi, "support": int(si)})
        log.info("rep %d done (qsvm acc %.3f, rbf acc %.3f)",
                 rep, (pred_q == yte).mean(), (pred_r == yte).mean())

    out = ROOT / "results" / "confusion"
    out.mkdir(parents=True, exist_ok=True)

    pc = pd.DataFrame(per_class_rows)
    pc_mean = pc.groupby(["class", "classifier"])[["precision", "recall", "f1", "support"]].mean().round(3)
    pc_mean.to_csv(out / "per_class_metrics.csv")

    for clf, cm in agg.items():
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                        fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{'QSVM' if clf == 'qsvm' else 'RBF-SVM'} — QUBO-8, aggregated over {n_rep} repeats")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{clf}.png", dpi=150)
        plt.close(fig)
        np.savetxt(out / f"confusion_{clf}.csv", cm, delimiter=",", fmt="%d",
                   header=",".join(classes), comments="")

    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f)
    print(pc_mean.to_string())
    print("saved to", out)


if __name__ == "__main__":
    main()
