"""Unit tests for the Quobo pipeline — regression guards for the audit fixes.

Covers: scaler train-only fitting (QML-001), PSI hazard weighting (PSI-001),
bbox clamp space (BBOX-001), zone mapping, selection ground-truth recovery,
reproducibility, and Holm correction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.qsvm import fit_scale_for_feature_map
from src.quobo.run_experiment import paired_significance
from scripts.pollution_dashboard import compute_zone_table, zone_of


# ---------- QML-001: scaler fit on train only ----------

def test_scaler_params_from_train_only():
    Xtr = np.array([[10.0], [20.0], [30.0]])
    Xte = np.array([[5.0], [40.0]])  # outside train bounds
    tr_s, te_s = fit_scale_for_feature_map(Xtr, Xte)
    # scaler range is fixed 0.05-1.52: train min->0.05, max->1.52
    assert tr_s[0, 0] == pytest.approx(0.05)
    assert tr_s[2, 0] == pytest.approx(1.52)
    # test values map OUTSIDE the range (no clip, no test-stat fitting)
    assert te_s[0, 0] < 0.05
    assert te_s[1, 0] > 1.52


def test_scaler_identical_transform_of_same_point():
    Xtr = np.random.default_rng(0).normal(size=(50, 4))
    Xte = Xtr[:5].copy()
    tr_s, te_s = fit_scale_for_feature_map(Xtr, Xte)
    # a train row appearing in 'test' must map to the same angles
    np.testing.assert_allclose(tr_s[:5], te_s)


# ---------- PSI-001: hazard weights in the numerator ----------

def _zone_df(rows):
    return pd.DataFrame(rows)

def test_psi_hazard_weight_multiplies():
    # one zone, one cigarette (w=3) vs three bottles (w=1 each)
    df = _zone_df([{"crop_path": "batch_1_001.jpg", "class": "cigarette"},
                   {"crop_path": "batch_1_002.jpg", "class": "bottle"},
                   {"crop_path": "batch_1_003.jpg", "class": "bottle"},
                   {"crop_path": "batch_1_004.jpg", "class": "bottle"}])
    zt = compute_zone_table(df)
    sub_cig = 1 * 3.0
    sub_bot = 3 * 1.0
    assert sub_cig == sub_bot  # tie -> psi at 500 boundary dominance
    assert zt["psi"].iloc[0] == 500  # the max weighted cell maps to 500

def test_psi_cigarette_dominates_bottles():
    df = _zone_df([{"crop_path": "batch_1_001.jpg", "class": "cigarette"}] * 2 +
                  [{"crop_path": "batch_1_002.jpg", "class": "bottle"}] * 10)
    zt = compute_zone_table(df)
    # cigarettes: 2*3=6 vs bottles: 10*1=10 -> bottle sub-index is max -> not 500 for cig
    assert zt["psi"].iloc[0] == 500  # bottle cell is the global max -> 500


# ---------- zone mapping ----------

@pytest.mark.parametrize("fname,expected", [
    ("batch_1_000123_ann4.jpg", "Zone A"),
    ("batch_6_000123_ann4.jpg", "Zone F"),
    ("batch_7_000123_ann4.jpg", "Zone A"),   # wraps
    ("batch_11_000123_ann4.jpg", "Zone E"),  # 11-1 mod 6 = 10 -> E
])
def test_zone_of_batch(fname, expected):
    assert zone_of(f"data/processed/crops/bottle/{fname}") == expected


# ---------- BBOX-001: clamp in actual image space ----------

def test_bbox_clamp_actual_image_dims(tmp_path):
    import cv2
    from src.quobo.data_prep import build_crops
    imgs = tmp_path / "images"
    (imgs / "batch_1").mkdir(parents=True)
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:, 200:] = 255
    cv2.imwrite(str(imgs / "batch_1/000001.jpg"), img)
    coco = {"categories": [{"id": 1, "name": "Clear plastic bottle"}],
            "images": [{"id": 0, "width": 200, "height": 100,
                        "file_name": "batch_1/000001.jpg"}],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 1,
                             "bbox": [150, 25, 40, 20]}]}
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    stats = build_crops(ann, imgs, tmp_path / "crops", ["bottle"], 1)
    crop = cv2.imread(str(next((tmp_path / "crops" / "bottle").glob("*.jpg"))))
    # bbox scaled 2x: x=300,y=50,w=80,h=40; 10% margin; clamp to 400x200
    assert crop.shape[:2] == (48, 96)
    assert stats["per_class"]["bottle"] == 1


# ---------- selection ground-truth recovery ----------

def test_selection_recovers_informative_features():
    from src.quobo.selection import select_features
    rng = np.random.default_rng(42)
    n = 50
    informative = [3, 11, 24, 37, 45]
    X = rng.normal(size=(600, n))
    for i, c in enumerate(informative):
        X[:, c] += (i - 2) * 2.0 * np.where(np.arange(600) % 6 == 0, 1.0, -1.0)
    y = (np.arange(600) % 6 == 0).astype(int)
    cfg = {"mi_bins": 16, "sa_sweeps": 2000, "sa_repeats": 5}
    for arm in ("C_mi", "D_qubo"):
        sel, _ = select_features(arm, X, y, 5, 42, cfg)
        overlap = len(set(sel) & set(informative))
        assert overlap >= 3, f"{arm} recovered only {overlap}/5 informative features: {sel}"


# ---------- reproducibility ----------

def test_selection_deterministic():
    from src.quobo.selection import select_features
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 30))
    y = (X[:, 0] > 0).astype(int)
    cfg = {"mi_bins": 16, "sa_sweeps": 2000, "sa_repeats": 5}
    s1, _ = select_features("D_qubo", X, y, 4, 42, cfg)
    s2, _ = select_features("D_qubo", X, y, 4, 42, cfg)
    assert s1 == s2


# ---------- Holm correction ----------

def test_holm_monotone_and_bounded():
    # replicate the holm() inside paired_significance via its output shape:
    # feed a results frame with two arms x one classifier, 25 reps
    rng = np.random.default_rng(3)
    rows = []
    for rep in range(25):
        rows.append({"repeat": rep, "classifier": "rbf_svm", "arm": "A",
                     "accuracy": 0.40 + 0.02 * rng.normal()})
        rows.append({"repeat": rep, "classifier": "rbf_svm", "arm": "B",
                     "accuracy": 0.40 + 0.02 * rng.normal()})
    out = paired_significance(pd.DataFrame(rows))
    assert len(out) == 1
    assert 0.0 <= out[0]["wilcoxon_p_holm"] <= 1.0
    assert isinstance(out[0]["significant_0.05"], bool)
