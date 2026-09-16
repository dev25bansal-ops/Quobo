"""Integration test: synthetic crops -> embeddings -> 4 arms -> classifiers.

Runs the real pipeline on ~96 synthetic images with a MobileNetV2-shaped
network (tiny random conv backbone, no ImageNet download) and asserts the
full flow works end-to-end and produces sane outputs. Target: < 2 min.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def synthetic_crops(tmp_path_factory):
    """96 tiny crops across 3 classes with a learnable color signal."""
    import cv2
    root = tmp_path_factory.mktemp("crops")
    rng = np.random.default_rng(0)
    classes = {"bottle": (40, 80, 200), "can": (200, 60, 40), "carton": (60, 200, 80)}
    for cls, base_rgb in classes.items():
        d = root / cls
        d.mkdir()
        for i in range(32):
            img = rng.integers(0, 40, size=(96, 96, 3), dtype=np.uint8)
            img[..., 0] += base_rgb[2]  # cv2 is BGR
            img[..., 1] += base_rgb[1]
            img[..., 2] += base_rgb[0]
            cv2.imwrite(str(d / f"{i:03d}.jpg"), img)
    return root, list(classes)


@pytest.mark.integration
def test_end_to_end_pipeline(synthetic_crops, monkeypatch, tmp_path):
    crops_dir, classes = synthetic_crops

    # tiny random-weight backbone instead of ImageNet MobileNetV2 — keeps the
    # real extract_embeddings code path (batching, preprocess, DataFrame)
    import tensorflow as tf

    from src.quobo import features as F

    def tiny_model():
        inp = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Conv2D(4, 8, strides=4, activation="relu")(inp)
        x = tf.keras.layers.Conv2D(8, 8, strides=4, activation="relu")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        return tf.keras.Model(inp, x)

    monkeypatch.setattr(F, "load_model", tiny_model)
    monkeypatch.setattr(F, "ROOT", tmp_path)
    # crops_fingerprint/paths are ROOT-relative: rebuild them under tmp
    proc = tmp_path / "data" / "processed"
    (proc).mkdir(parents=True)
    import shutil
    shutil.copytree(crops_dir, proc / "crops")

    cfg = {
        "seed": 42,
        "data": {"classes": classes, "min_images_per_class": 5, "img_size": 224},
        "features": {"backbone": "tiny", "pca_components": 6, "standardize": True},
        "selection": {"k": 4},
        "qubo": {"mi_bins": 8, "sa_sweeps": 1500, "sa_repeats": 5},
        "qsvm": {"reps": 1, "entanglement": "linear",
                 "max_train_samples": 48, "test_fraction": 0.25},
        "experiment": {"n_repeats": 2, "arms": ["A_random", "B_pca", "C_mi", "D_qubo"]},
    }

    df = F.run_features(cfg)
    assert len(df) == 96
    assert df["label"].nunique() == 3

    # a color-signal dataset must be separable: quick sanity via the real arms
    from src.quobo.qsvm import run_all_classifiers
    from src.quobo.selection import select_features
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X, y = df[feat_cols].values.astype(float), np.asarray(df["label"].values)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    sel, _ = select_features("D_qubo", Xtr, ytr, 4, 42, cfg["qubo"])
    assert len(sel) == 4
    assert all(0 <= s < X.shape[1] for s in sel)
    metrics = run_all_classifiers(Xtr[:, sel], ytr, Xte[:, sel], yte, cfg, rep_offset=0)
    # strong color signal -> classifiers must beat chance (33%) decisively
    for clf, m in metrics.items():
        assert m["accuracy"] > 0.6, f"{clf} only {m['accuracy']:.2f} on separable data"
    # determinism: same inputs -> same accuracy
    metrics2 = run_all_classifiers(Xtr[:, sel], ytr, Xte[:, sel], yte, cfg, rep_offset=0)
    assert metrics["qsvm"]["accuracy"] == metrics2["qsvm"]["accuracy"]
