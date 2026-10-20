"""Unit test for data_prep: bounding-box clamp in actual image space (BBOX-001)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_bbox_clamp_actual_image_dims(tmp_path):
    import cv2

    from src.quobo.data_prep import build_crops

    imgs = tmp_path / "images"
    (imgs / "batch_1").mkdir(parents=True)
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:, 200:] = 255
    cv2.imwrite(str(imgs / "batch_1/000001.jpg"), img)
    coco = {
        "categories": [{"id": 1, "name": "Clear plastic bottle"}],
        "images": [{"id": 0, "width": 200, "height": 100, "file_name": "batch_1/000001.jpg"}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [150, 25, 40, 20]}],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    stats = build_crops(ann, imgs, tmp_path / "crops", ["bottle"], 1)
    crop = cv2.imread(str(next((tmp_path / "crops" / "bottle").glob("*.jpg"))))
    # bbox scaled 2x: x=300,y=50,w=80,h=40; 10% margin; clamp to 400x200
    assert crop.shape[:2] == (48, 96)
    assert stats["per_class"]["bottle"] == 1


def _write_taxonomy_csv(path, rows):
    path.write_text(
        "leaf,coarse\n" + "\n".join(f"{leaf},{coarse}" for leaf, coarse in rows) + "\n",
        encoding="utf-8",
    )


def _write_annotations(path, cats):
    path.write_text(json.dumps({"categories": cats, "images": [], "annotations": []}), "utf-8")


def test_taxonomy_drift_reports_presence_drift(tmp_path):
    """L-10: a leaf missing from the CSV (or unknown to TACO) is reported."""
    from src.quobo.data_prep import check_taxonomy_drift

    csv_path = tmp_path / "taxonomy.csv"
    ann_path = tmp_path / "annotations.json"
    _write_taxonomy_csv(csv_path, [("Bottle", "bottle"), ("Ghost leaf", "other")])
    _write_annotations(
        ann_path,
        [
            {"id": 0, "name": "Bottle", "supercategory": "Bottle"},
            {"id": 1, "name": "Can", "supercategory": "Can"},
        ],
    )

    report = check_taxonomy_drift(csv_path, ann_path)
    assert report["annotations_available"] is True
    assert report["missing_in_csv"] == ["Can"]
    assert report["missing_in_taco"] == ["Ghost leaf"]
    assert report["ok"] is False


def test_taxonomy_drift_reports_coarse_mismatch(tmp_path):
    """L-10: a CSV coarse value that disagrees with TACO's supercategory is reported."""
    from src.quobo.data_prep import check_taxonomy_drift

    csv_path = tmp_path / "taxonomy.csv"
    ann_path = tmp_path / "annotations.json"
    # "bottle" is a configured coarse class here (some leaf maps to it), so
    # TACO's supercategory "Bottle" implies "bottle" — and the CSV's "other"
    # for that leaf is the disagreement.
    _write_taxonomy_csv(
        csv_path, [("Clear plastic bottle", "bottle"), ("Bottle", "other"), ("Can", "can")]
    )
    _write_annotations(
        ann_path,
        [
            {"id": 0, "name": "Clear plastic bottle", "supercategory": "Bottle"},
            {"id": 1, "name": "Bottle", "supercategory": "Bottle"},
            {"id": 2, "name": "Can", "supercategory": "Can"},
        ],
    )

    report = check_taxonomy_drift(csv_path, ann_path)
    leaves = {m["leaf"] for m in report["coarse_mismatches"]}
    assert "Bottle" in leaves
    assert report["missing_in_csv"] == []
    assert report["missing_in_taco"] == []


def test_taxonomy_drift_detects_group_inconsistency(tmp_path):
    """L-10: one supercategory split across two coarse classes is hard drift."""
    from src.quobo.data_prep import check_taxonomy_drift

    csv_path = tmp_path / "taxonomy.csv"
    ann_path = tmp_path / "annotations.json"
    _write_taxonomy_csv(csv_path, [("Drink can", "can"), ("Aerosol", "other")])
    _write_annotations(
        ann_path,
        [
            {"id": 0, "name": "Drink can", "supercategory": "Can"},
            {"id": 1, "name": "Aerosol", "supercategory": "Can"},
        ],
    )

    report = check_taxonomy_drift(csv_path, ann_path)
    assert [g["leaf"] for g in report["group_inconsistencies"]] == ["Aerosol"]
    # a grouping refinement is reported but is not "hard" drift: both leaves
    # are still mapped to a real coarse class, so no crop is lost.
    assert report["n_soft_drift"] >= 1
    assert report["n_hard_drift"] == 0


def test_taxonomy_drift_degrades_gracefully_without_annotations(tmp_path):
    """L-10: a fresh clone has no annotations.json — the check must not crash."""
    from src.quobo.data_prep import check_taxonomy_drift

    csv_path = tmp_path / "taxonomy.csv"
    _write_taxonomy_csv(csv_path, [("Bottle", "bottle")])

    report = check_taxonomy_drift(csv_path, tmp_path / "does_not_exist.json")
    assert report["annotations_available"] is False
    assert "not found" in report["message"]
    assert report["missing_in_csv"] == []
    assert report["n_csv_leaves"] == 1


def test_shipped_taxonomy_matches_taco_annotations():
    """L-10: the shipped CSV must not drift from the shipped annotations.

    This is the regression guard for the hand-maintained CSV: it fails loudly
    if TACO's category list changes without the CSV being updated.
    """
    from src.quobo.config import ROOT
    from src.quobo.data_prep import ANNOTATIONS_REL_PATH, check_taxonomy_drift

    ann_path = ROOT / ANNOTATIONS_REL_PATH
    if not ann_path.is_file():
        pytest.skip("TACO annotations.json not present (dataset not downloaded)")
    report = check_taxonomy_drift()
    assert report["missing_in_csv"] == [], (
        "the taxonomy CSV does not cover every TACO leaf — those crops are "
        "silently dropped by build_crops"
    )
    assert report["missing_in_taco"] == []
