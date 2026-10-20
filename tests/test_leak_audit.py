"""Leak-audit tests — the auditor must be able to FAIL, not merely pass.

The audit is only worth having if it can detect the leaks it claims to
detect. Every test here plants a specific defect in a synthetic run and
asserts the audit notices it. A test that only ever feeds the auditor a
clean run proves nothing.

Offline: no network, no model, no qiskit, no training.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.leak_audit as CLI
import src.quobo.leak_audit as LA
import src.quobo.run_experiment as RE

# --------------------------------------------------------------------------- #
# Fixture: a synthetic, COMPLETE, leak-free run on disk
# --------------------------------------------------------------------------- #

ARMS = ["A_random", "D_qubo"]
CLASSES = ["bottle", "can"]


def _crop_name(group: str, idx: int) -> str:
    return f"{group}_ann{idx:04d}.jpg"


def _build_run(
    root: Path,
    *,
    reps: int = 2,
    straddle: bool = False,
    duplicate_ids: bool = False,
    misalign_y: bool = False,
    misalign_preds: bool = False,
    tamper: bool = False,
    split_differs: bool = False,
    group_label_mismatch: bool = False,
    missing_crop: bool = False,
    manifest_status: str = "complete",
    manifest_overlap: int | None = None,
) -> Path:
    """Build a run directory under ``root/experiments/run_x``.

    ``root`` is the "repo" root: crops live in ``root/data/processed/crops``
    and the run in ``root/experiments/run_x``, mirroring the real layout.
    """
    crops_root = root / "data" / "processed" / "crops"
    for cls in CLASSES:
        (crops_root / cls).mkdir(parents=True, exist_ok=True)

    # 4 photos per class, 2 crops each => 8 crops per class, 16 total
    photos: dict[str, list[str]] = {}
    for cls in CLASSES:
        for p in range(4):
            group = f"{cls}_photo{p}"
            names = []
            for j in range(2):
                name = _crop_name(group, j)
                (crops_root / cls / name).write_bytes(b"not-a-real-jpeg")
                names.append(name)
            photos[group] = names

    all_groups = list(photos)
    run_dir = root / "experiments" / "run_x"
    run_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(reps):
        # Each repeat holds out an overlapping window of photos — a real run
        # reuses the same photo pool with different splits, so consecutive
        # repeats share photos (that is NOT leakage).
        start = rep % max(1, len(all_groups) - 2)
        held_groups = all_groups[start : start + 3] or all_groups[:2]
        test_ids = []
        y_test = []
        for g in held_groups:
            for name in photos[g]:
                test_ids.append(str(crops_root / g.split("_photo")[0] / name))
                y_test.append(g.split("_photo")[0])
        if duplicate_ids and test_ids:
            test_ids.append(test_ids[0])
            y_test.append(y_test[0])
        if misalign_y:
            y_test = y_test[:-1]
        groups = [_group_for(t) for t in test_ids]
        if group_label_mismatch:
            groups = ["wrong_group" for _ in groups]

        for arm in ARMS:
            ids = list(test_ids)
            if split_differs and arm == ARMS[-1] and ids:
                ids = ids[:-1]
            arm_groups = groups if len(groups) == len(ids) else [_group_for(t) for t in ids]
            payload = {
                "arm": arm,
                "rep": rep,
                "test_ids": ids,
                "y_test": list(y_test),
                "test_groups": arm_groups,
                "metrics": {
                    "rbf_svm": {
                        "accuracy": 0.5,
                        "predictions": ["bottle"] * (len(ids) - 1 if misalign_preds else len(ids)),
                    }
                },
            }
            (run_dir / f"rep{rep}_{arm}.json").write_text(json.dumps(payload), encoding="utf-8")

    # a photo whose crops straddle the split: one crop held out, one in train
    if straddle:
        cls = CLASSES[0]
        group = "bottle_photo0"
        payload_path = run_dir / "rep0_A_random.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        held = [t for t in payload["test_ids"] if group in t]
        if held:
            # drop one crop of the held-out photo from the held-out set:
            # that crop now sits on the train side => a straddling photo
            payload["test_ids"] = [t for t in payload["test_ids"] if t != held[0]]
            payload["y_test"] = payload["y_test"][: len(payload["test_ids"])]
            payload["test_groups"] = [_group_for(t) for t in payload["test_ids"]]
            payload["metrics"]["rbf_svm"]["predictions"] = ["bottle"] * len(payload["test_ids"])
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

    if missing_crop:
        # delete a crop that the artifacts still reference
        victim = sorted(crops_root.rglob("*.jpg"))[0]
        victim.unlink()

    # the manifest, with a real sha256 inventory
    artifacts = {}
    for path in sorted(run_dir.glob("rep*.json")):
        artifacts[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    if tamper:
        target = run_dir / "rep0_A_random.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["metrics"]["rbf_svm"]["accuracy"] = 0.99  # rewrite, hash now differs
        target.write_text(json.dumps(payload), encoding="utf-8")

    manifest = {
        "run_id": "run_x",
        "status": manifest_status,
        "reps_completed": reps,
        "arms": list(ARMS),
        "artifacts": artifacts,
        "seed": 42,
    }
    if manifest_overlap is not None:
        manifest["group_overlap_events"] = manifest_overlap
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _group_for(crop_id: str) -> str:
    return str(crop_id).replace("\\", "/").split("/")[-1].rsplit("_ann", 1)[0]


@pytest.fixture()
def clean_run(tmp_path: Path) -> Path:
    return _build_run(tmp_path)


# --------------------------------------------------------------------------- #
# The happy path, and the guard against a vacuous auditor
# --------------------------------------------------------------------------- #


def test_clean_run_passes(clean_run: Path):
    audit = LA.audit_run(clean_run)
    assert audit.ok, LA.format_report(audit)
    assert audit.status == "complete"
    assert len(audit.repeats) == 2
    assert audit.straddle_check_performed is True


def test_audit_reports_repeats_sharing_photos(clean_run: Path):
    """Repeats reuse the photo pool, so overlap BETWEEN repeats is expected.

    This is NOT leakage — a photo held out in two different repeats is simply
    the same photo appearing in two different splits. It is reported so a
    reader can see the repeats are genuinely different splits rather than
    one split repeated 25 times.
    """
    audit = LA.audit_run(clean_run)
    assert audit.repeats_sharing_photos > 0
    assert audit.ok  # sharing photos across repeats is not leakage


def test_straddling_photo_is_detected(tmp_path: Path):
    """The core leak: one source photo on both sides of a split."""
    run_dir = _build_run(tmp_path, straddle=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert sum(r.straddling_groups for r in audit.repeats) >= 1
    assert "straddling" in LA.format_report(audit)


def test_duplicate_test_ids_are_detected(tmp_path: Path):
    run_dir = _build_run(tmp_path, duplicate_ids=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert any(r.duplicate_test_ids > 0 for r in audit.repeats)


def test_misaligned_y_test_is_detected(tmp_path: Path):
    run_dir = _build_run(tmp_path, misalign_y=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert any(r.misaligned_artifacts > 0 for r in audit.repeats)


def test_misaligned_predictions_are_detected(tmp_path: Path):
    run_dir = _build_run(tmp_path, misalign_preds=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert any(r.misaligned_artifacts > 0 for r in audit.repeats)


def test_tampered_artifact_is_detected(tmp_path: Path):
    """A substituted file must fail the hash check."""
    run_dir = _build_run(tmp_path, tamper=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert any(r.hash_mismatches > 0 for r in audit.repeats)


def test_arms_with_different_splits_are_detected(tmp_path: Path):
    """Paired significance is invalid if the arms saw different crops."""
    run_dir = _build_run(tmp_path, split_differs=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert any(r.arms_with_different_split for r in audit.repeats)


def test_group_label_mismatch_is_detected(tmp_path: Path):
    run_dir = _build_run(tmp_path, group_label_mismatch=True)
    audit = LA.audit_run(run_dir)
    assert not audit.ok
    assert any(r.group_label_mismatches > 0 for r in audit.repeats)


def test_missing_crop_is_reported(tmp_path: Path):
    """A crop referenced by the artifacts but gone from disk is a hole."""
    run_dir = _build_run(tmp_path, missing_crop=True)
    audit = LA.audit_run(run_dir)
    assert any(r.held_out_crops_missing > 0 for r in audit.repeats)


def test_missing_crop_inventory_is_not_a_silent_pass(tmp_path: Path):
    """Without the inventory the straddle check cannot run — say so, do not pass."""
    run_dir = _build_run(tmp_path)
    audit = LA.audit_run(run_dir, crops_root=tmp_path / "does" / "not" / "exist")
    assert audit.straddle_check_performed is False
    assert not audit.ok
    assert any("crop inventory" in e for e in audit.errors)


# --------------------------------------------------------------------------- #
# Refusals: an unauditable run is not evidence
# --------------------------------------------------------------------------- #


def test_incomplete_run_is_refused(tmp_path: Path):
    run_dir = _build_run(tmp_path, manifest_status="running")
    with pytest.raises(LA.LeakAuditError):
        LA.audit_run(run_dir)


def test_failed_run_is_refused(tmp_path: Path):
    run_dir = _build_run(tmp_path, manifest_status="failed")
    with pytest.raises(LA.LeakAuditError):
        LA.audit_run(run_dir)


def test_missing_manifest_is_refused(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    (run_dir / "run_manifest.json").unlink()
    with pytest.raises(LA.LeakAuditError):
        LA.audit_run(run_dir)


def test_corrupt_manifest_is_refused(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    (run_dir / "run_manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(LA.LeakAuditError):
        LA.audit_run(run_dir)


def test_zero_repeats_is_refused(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["reps_completed"] = 0
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(LA.LeakAuditError):
        LA.audit_run(run_dir)


# --------------------------------------------------------------------------- #
# Manifest agreement
# --------------------------------------------------------------------------- #


def test_manifest_overlap_agreement_is_reported(tmp_path: Path):
    run_dir = _build_run(tmp_path, manifest_overlap=0)
    audit = LA.audit_run(run_dir)
    assert audit.manifest_group_overlap_events == 0
    assert audit.manifest_overlap_agrees is True
    assert audit.ok


def test_manifest_overlap_disagreement_fails_the_audit(tmp_path: Path):
    """A manifest that claims 0 overlaps while the artifacts show one must fail."""
    run_dir = _build_run(tmp_path, straddle=True, manifest_overlap=0)
    audit = LA.audit_run(run_dir)
    assert audit.manifest_overlap_agrees is False
    assert not audit.ok
    assert any("disagree" in e for e in audit.errors)


def test_run_without_overlap_counter_is_reported_as_unrecorded(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    audit = LA.audit_run(run_dir)
    assert audit.manifest_group_overlap_events is None
    assert "not recorded by this run" in LA.format_report(audit)


# --------------------------------------------------------------------------- #
# The auditor's own grouping must not drift from the pipeline's
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "crop_id",
    [
        "D:\\Quobo\\data\\processed\\crops\\bottle\\batch_10_000010_ann375.jpg",
        "batch_3_000123_ann45.jpg",
        "/abs/path/carton_9_ann1.jpg",
    ],
)
def test_group_of_matches_pipeline_grouping(crop_id: str):
    """The audit deliberately re-implements the grouping; it must agree.

    If these two ever diverge, the audit is reasoning about a different
    grouping than the pipeline produced — and would silently under-report.
    """
    assert LA._group_of(crop_id) == RE.crop_image_group(crop_id)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_exit_codes(tmp_path: Path, monkeypatch, capsys):
    run_dir = _build_run(tmp_path)
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    assert CLI.main(["--run-id", "run_x"]) == 0
    assert "PASS" in capsys.readouterr().out

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "running"
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert CLI.main(["--run-id", "run_x"]) == 2  # not auditable


def test_cli_json_output(tmp_path: Path, monkeypatch, capsys):
    _build_run(tmp_path)
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    assert CLI.main(["--run-id", "run_x", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["n_repeats_audited"] == 2


def test_cli_rejects_traversal(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    assert CLI.main(["--run-id", "../../etc"]) == 2
    err = capsys.readouterr().err  # read once: a second call returns empty
    assert ".." in err
    assert "leak audit failed" in err


def test_cli_rejects_absolute_run_id(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    assert CLI.main(["--run-id", "C:/Windows/System32"]) == 2
    assert "leak audit failed" in capsys.readouterr().err


def test_cli_finds_latest_run(tmp_path: Path, monkeypatch, capsys):
    _build_run(tmp_path)
    monkeypatch.setattr(CLI, "ROOT", tmp_path)
    assert CLI.main([]) == 0
    assert "run_x" in capsys.readouterr().out
