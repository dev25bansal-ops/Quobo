"""Q04/Q05 provenance guards — offline, no network, no model, no cloud.

Covers:
  * run_experiment persists exact held-out crop IDs aligned with labels/preds
  * run manifest lifecycle: 'running' -> 'complete' (or 'failed')
  * dashboard consumes saved IDs (never replays a split) and rejects
    incomplete, legacy, tampered and misaligned runs
  * explicit prediction vs demo-ground-truth modes with no truth fallback
  * explicit empty-data handling in the loader and renderer
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.pollution_dashboard as PD
import src.quobo.run_experiment as RE

# --------------------------------------------------------------------------- #
# Offline fakes for run_experiment.run (no embeddings, no qiskit, no sklearn)
# --------------------------------------------------------------------------- #


def _valid_cfg() -> dict:
    return {
        "seed": 7,
        "data": {"classes": ["bottle", "cigarette"], "min_images_per_class": 10},
        "features": {"backbone": "MobileNetV2", "pca_components": 10},
        "selection": {"k": 2},
        "qubo": {"mi_bins": 4, "sa_sweeps": 1000, "sa_repeats": 5},
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 50,
            "test_fraction": 0.25,
            "tune_rbf": True,
            "max_qsvm_features": 20,
        },
        "experiment": {"n_repeats": 5, "arms": ["D_qubo"]},
    }


def _raw_frame(n_groups: int = 8, per_group: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for g in range(n_groups):
        label = "bottle" if g % 2 == 0 else "cigarette"
        for j in range(per_group):
            row = {f"e{i}": float(rng.normal()) for i in range(12)}
            row["label"] = label
            # crop_image_group() strips the _ann suffix -> group "batch_{g}_0001"
            row["crop_path"] = f"/crops/{label}/batch_{g}_0001_ann{j}.jpg"
            rows.append(row)
    return pd.DataFrame(rows)


def _fake_pca(Xtr, Xall, n_comp, seed):
    return Xtr[:, :n_comp], Xall[:, :n_comp], 99.0


def _fake_select(arm, Xtr, ytr, k, seed, qubo):
    return list(range(min(k, Xtr.shape[1]))), {"I": [0.0] * Xtr.shape[1]}


def _fake_classifiers(Xtr, ytr, Xte, yte, cfg, rep_offset=0):
    return {
        "rbf_svm": {
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "predictions": [str(v) for v in yte],
        }
    }


def _patch_run(monkeypatch, tmp_path) -> pd.DataFrame:
    monkeypatch.setattr(RE, "ROOT", tmp_path)
    (tmp_path / "results" / "tables").mkdir(parents=True)
    raw = _raw_frame()
    monkeypatch.setattr(RE, "run_raw_embeddings", lambda cfg: raw)
    monkeypatch.setattr(RE, "fit_pca_on_train", _fake_pca)
    monkeypatch.setattr(RE, "select_features", _fake_select)
    monkeypatch.setattr(RE, "run_all_classifiers", _fake_classifiers)
    return raw


# --------------------------------------------------------------------------- #
# Q04: persisted IDs + manifest lifecycle
# --------------------------------------------------------------------------- #


def test_run_persists_aligned_test_ids_and_completes_manifest(tmp_path, monkeypatch):
    raw = _patch_run(monkeypatch, tmp_path)
    label_by_path = dict(zip(raw["crop_path"], raw["label"]))
    seen = {}

    def capturing(*args, **kwargs):
        manifests = list((tmp_path / "experiments").glob(f"*/{RE.MANIFEST_NAME}"))
        assert len(manifests) == 1
        seen["status"] = json.loads(manifests[0].read_text(encoding="utf-8"))["status"]
        return _fake_classifiers(*args, **kwargs)

    monkeypatch.setattr(RE, "run_all_classifiers", capturing)

    results = RE.run(_valid_cfg(), run_id="run_test", config_path="configs/x.yaml")
    run_dir = tmp_path / "experiments" / "run_test"
    manifest = json.loads((run_dir / RE.MANIFEST_NAME).read_text(encoding="utf-8"))

    # manifest was 'running' while artifacts were still being produced
    assert seen["status"] == "running"
    assert manifest["status"] == "complete"
    assert manifest["reps_completed"] == 5
    assert manifest["config_sha256"]
    assert results.shape[0] == 5  # 1 arm x 1 classifier x 5 repeats

    for rep in range(5):
        name = f"rep{rep}_D_qubo.json"
        payload = json.loads((run_dir / name).read_text(encoding="utf-8"))
        ids = payload["test_ids"]
        preds = payload["metrics"]["rbf_svm"]["predictions"]
        # exact 1:1 alignment across IDs, true labels and predictions
        assert len(ids) == len(payload["y_test"]) == len(preds)
        assert len(set(ids)) == len(ids)
        assert [label_by_path[i] for i in ids] == payload["y_test"]
        assert payload["test_groups"]
        assert name in manifest["artifacts"]
        assert manifest["artifacts"][name]["sha256"]


def test_failed_run_manifest_is_marked_failed(tmp_path, monkeypatch):
    _patch_run(monkeypatch, tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("classifier exploded")

    monkeypatch.setattr(RE, "run_all_classifiers", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        RE.run(_valid_cfg(), run_id="run_fail")
    manifest = json.loads(
        (tmp_path / "experiments" / "run_fail" / RE.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert "exploded" in manifest["error"]


# --------------------------------------------------------------------------- #
# Q05: dashboard helpers
# --------------------------------------------------------------------------- #


def _write_run(
    root: Path,
    run_id: str = "run_ok",
    arm: str = "D_qubo",
    reps: int = 2,
    status: str = "complete",
    ids=None,
    y_test=None,
    preds=None,
    tamper: bool = False,
    drop_artifact: bool = False,
) -> Path:
    """Build a minimal manifest+reps run fixture under ``root/experiments``."""
    run_dir = root / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for rep in range(reps):
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "rep": rep,
            "arm": arm,
            "seed": 1 + rep,
            "y_test": y_test if y_test is not None else ["bottle", "cigarette"],
            "test_ids": ids
            if ids is not None
            else [f"saved_{rep}_a_ann0.jpg", f"saved_{rep}_b_ann1.jpg"],
            "metrics": {
                "rbf_svm": {"predictions": preds if preds is not None else ["bottle", "cigarette"]}
            },
        }
        path = run_dir / f"rep{rep}_{arm}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifacts[path.name] = {"sha256": PD._sha256_file(path), "bytes": path.stat().st_size}
    if drop_artifact:
        artifacts.pop(f"rep{reps - 1}_{arm}.json", None)
        (run_dir / f"rep{reps - 1}_{arm}.json").unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "implementation": "run-manifest-v1",
        "run_id": run_id,
        "status": status,
        "arms": [arm],
        "reps_completed": reps,
        "artifacts": artifacts,
    }
    (run_dir / PD.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    if tamper:
        (run_dir / f"rep0_{arm}.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_dashboard_consumes_saved_ids_not_replayed_split(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, ids=["saved_only_A_ann0.jpg", "saved_only_B_ann1.jpg"])
    df = PD.load_predictions()
    # the saved IDs are the authority — no data dir / split replay is consulted
    assert set(df["crop_path"]) == {"saved_only_A_ann0.jpg", "saved_only_B_ann1.jpg"}
    assert df.attrs["run_id"] == "run_ok"
    assert df.attrs["mode"] == "predictions"
    classes = dict(zip(df["crop_path"], df["class"]))
    assert classes["saved_only_A_ann0.jpg"] == "bottle"
    assert classes["saved_only_B_ann1.jpg"] == "cigarette"


def test_legacy_artifact_without_ids_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    run_dir = _write_run(tmp_path, run_id="legacy")
    path = run_dir / "rep0_D_qubo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["test_ids"]  # simulate a pre-Q04 artifact
    path.write_text(json.dumps(payload), encoding="utf-8")
    # keep the manifest hash honest so the failure is about IDs, not corruption
    manifest_path = run_dir / PD.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][path.name] = {
        "sha256": PD._sha256_file(path),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PD.ProvenanceError, match="test_ids"):
        PD.load_predictions()


def test_misaligned_predictions_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, preds=["bottle"])  # 1 prediction vs 2 ids/labels
    with pytest.raises(PD.ProvenanceError, match="misaligned"):
        PD.load_predictions()


def test_incomplete_manifest_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, status="running")
    with pytest.raises(PD.ProvenanceError, match="not 'complete'"):
        PD.load_predictions(run_id="run_ok")
    # the scan path also refuses, it just reports there is no complete run
    with pytest.raises(PD.ProvenanceError, match="no completed experiment run"):
        PD.load_predictions()


def test_missing_artifact_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, reps=2, drop_artifact=True)
    with pytest.raises(PD.ProvenanceError, match="missing artifact"):
        PD.load_predictions()


def test_tampered_artifact_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, tamper=True)
    with pytest.raises(PD.ProvenanceError, match="manifest"):
        PD.load_predictions()


def test_no_run_does_not_fall_back_to_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    with pytest.raises(PD.ProvenanceError, match="no experiments directory"):
        PD.load_predictions()


def test_empty_prediction_run_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, reps=1, ids=[], y_test=[], preds=[])
    with pytest.raises(ValueError, match="empty dataset"):
        PD.load_predictions()


def test_ground_truth_requires_explicit_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    with pytest.raises(PD.ProvenanceError, match="ground-truth crops"):
        PD.load_crops()
    with pytest.raises(PD.ProvenanceError, match="ground-truth crops"):
        PD.main(["--mode", "demo-ground-truth"])
    # prediction mode still fails explicitly rather than substituting truth
    with pytest.raises(PD.ProvenanceError, match="no experiments directory"):
        PD.main([])


def test_render_empty_zone_table_is_explicit(tmp_path):
    empty = PD.compute_zone_table(pd.DataFrame(columns=["crop_path", "class"]))
    out = tmp_path / "dashboard.html"
    PD.render_dashboard(empty, out)
    assert "No observations" in out.read_text(encoding="utf-8")


def test_main_prediction_mode_renders_run_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "ROOT", tmp_path)
    _write_run(tmp_path, run_id="run_ok", reps=2)
    PD.main([])
    html = (tmp_path / "results" / "dashboard" / "dashboard.html").read_text(encoding="utf-8")
    assert "run_ok" in html
    assert "No observations" not in html
