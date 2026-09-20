"""M-07: tests for scripts/confusion_analysis.py — the per-class confusion
matrices and per-class metrics feed the paper's figures, so the aggregation
logic is covered here (it previously had no tests at all).

The script consumes persisted per-repeat prediction JSONs, so these tests
synthesise a run directory and assert on the produced artifacts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import confusion_analysis as CA


def _write_rep(run_dir: Path, rep: int, classes, y_test, qsvm_pred, rbf_pred):
    p = run_dir / f"rep{rep:02d}_D_qubo.json"
    p.write_text(
        json.dumps(
            {
                "rep": rep,
                "classes": list(classes),
                "y_test": [str(v) for v in y_test],
                "metrics": {
                    "qsvm": {"predictions": [str(v) for v in qsvm_pred]},
                    "rbf_svm": {"predictions": [str(v) for v in rbf_pred]},
                },
            }
        ),
        encoding="utf-8",
    )
    return p


def _write_manifest(run_dir: Path, reps, tamper: str | None = None):
    """Write a run_manifest whose artifacts hash the rep files. Passing
    ``tamper`` records a wrong hash for that file to exercise rejection."""
    import hashlib

    artifacts = {}
    for rep in reps:
        h = hashlib.sha256()
        h.update(rep.read_bytes())
        artifacts[rep.name] = {"sha256": h.hexdigest(), "bytes": rep.stat().st_size}
    if tamper is not None:
        artifacts[tamper]["sha256"] = "0" * 64
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _make_run(tmp_path, classes, n_reps=3, seed=0):
    run_dir = tmp_path / "experiments" / "20260101_120000"
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    written = []
    for rep in range(n_reps):
        # string labels, exactly like run_experiment persists
        y = rng.choice(classes, size=40)
        q = y.copy()
        r = y.copy()
        for i in np.where(rng.random(len(y)) < 0.125)[0]:  # rbf flips 1 in 8
            r[i] = classes[(classes.index(y[i]) + 1) % len(classes)]
        written.append(_write_rep(run_dir, rep, classes, y, q, r))
    _write_manifest(run_dir, written)
    return run_dir


def test_confusion_aggregation_and_artifacts(tmp_path, monkeypatch):
    classes = ["bottle", "can", "cup"]
    # run dir discovered by CA.main() through the ROOT monkeypatch below
    _make_run(tmp_path, classes, n_reps=3, seed=1)
    monkeypatch.setattr(CA, "ROOT", tmp_path)

    out = tmp_path / "results" / "confusion"
    CA.main()

    # QSVM was perfect in the fixture -> identity confusion matrix
    cm = np.loadtxt(out / "confusion_qsvm.csv", delimiter=",", skiprows=1, dtype=int)
    assert cm.shape == (3, 3)
    assert cm.sum() == 3 * 40  # 3 reps x 40 samples
    assert np.all(cm - np.diag(cm.diagonal()) == 0)  # zero off-diagonal

    # RBF flipped ~1/8 -> off-diagonal entries exist but diagonal dominates
    cm_r = np.loadtxt(out / "confusion_rbf_svm.csv", delimiter=",", skiprows=1, dtype=int)
    assert cm_r.sum() == 3 * 40
    assert np.all(cm_r.diagonal() > 0)

    # per-class metrics: qsvm precision/recall == 1.0
    pc = pd.read_csv(out / "per_class_metrics.csv")
    assert set(pc["classifier"]) == {"qsvm", "rbf_svm"}
    assert set(pc["class"]) == set(classes)
    q = pc[pc["classifier"] == "qsvm"]
    assert q["precision"].max() == pytest.approx(1.0)
    assert q["recall"].min() == pytest.approx(1.0)
    r = pc[pc["classifier"] == "rbf_svm"]
    assert r["recall"].max() < 1.0  # the flips cost recall


def test_latest_run_dir_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(CA, "ROOT", tmp_path)
    exp = tmp_path / "experiments"
    for name in ("20260101_000000", "20260202_000000"):
        (exp / name).mkdir(parents=True)
    assert CA.latest_run_dir().name == "20260202_000000"


def test_no_runs_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(CA, "ROOT", tmp_path)
    (tmp_path / "experiments").mkdir()
    with pytest.raises(SystemExit, match="no experiment runs"):
        CA.latest_run_dir()


def test_tampered_rep_is_rejected(tmp_path, monkeypatch):
    """Sec-note: a rep file whose hash no longer matches the manifest is refused."""
    classes = ["bottle", "can", "cup"]
    run_dir = _make_run(tmp_path, classes, n_reps=2, seed=3)
    monkeypatch.setattr(CA, "ROOT", tmp_path)

    # rewrite the manifest with a wrong hash for the first rep
    reps = sorted(run_dir.glob("rep*_D_qubo.json"))
    _write_manifest(run_dir, reps, tamper=reps[0].name)

    with pytest.raises(SystemExit, match="does not match the run manifest"):
        CA.main()


def test_missing_manifest_is_rejected(tmp_path, monkeypatch):
    classes = ["bottle", "can", "cup"]
    _make_run(tmp_path, classes, n_reps=2, seed=4)
    monkeypatch.setattr(CA, "ROOT", tmp_path)
    (tmp_path / "experiments" / "20260101_120000" / "run_manifest.json").unlink()

    with pytest.raises(SystemExit, match="no run_manifest.json"):
        CA.main()
