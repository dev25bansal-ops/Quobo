"""Stage 2: feature extraction — MobileNetV2 embeddings -> PCA -> features.csv.

Each crop is resized to 224x224, passed through MobileNetV2 (ImageNet weights,
global-average-pooled -> 1280-d), then PCA compresses to N candidate features.
Output: data/features/features.csv with columns f0..f{N-1}, label, crop_path.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import ROOT

log = logging.getLogger(__name__)


def load_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    return base


def extract_embeddings(model, crops_dir: Path, classes: list[str]) -> pd.DataFrame:
    paths, labels = [], []
    for cls in classes:
        cls_dir = crops_dir / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*.jpg")):
            paths.append(str(p))
            labels.append(cls)

    embs = np.zeros((len(paths), 1280), dtype=np.float32)
    batch, bs = 0, 64
    for i in tqdm(range(0, len(paths), bs), desc="embeddings"):
        chunk = paths[i : i + bs]
        imgs = np.stack(
            [tf.keras.utils.load_img(p, target_size=(224, 224)) for p in chunk]
        )
        imgs = tf.keras.applications.mobilenet_v2.preprocess_input(imgs)
        embs[i : i + bs] = model.predict(imgs, verbose=0)
    return pd.DataFrame(embs, columns=[f"e{i}" for i in range(1280)]).assign(
        label=labels, crop_path=paths
    )


def run_features(cfg: dict) -> pd.DataFrame:
    crops_dir = ROOT / "data" / "processed" / "crops"
    classes = cfg["data"]["classes"]
    tag = "features" if len(classes) == 7 else f"features_{len(classes)}class"
    out_csv = ROOT / "data" / "features" / f"{tag}.csv"

    if out_csv.exists():
        log.info("features exist: %s", out_csv)
        return pd.read_csv(out_csv)

    model = load_model()
    df = extract_embeddings(model, crops_dir, classes)

    X = df[[c for c in df.columns if c.startswith("e")]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = cfg["features"]["pca_components"]
    pca = PCA(n_components=n_comp, random_state=cfg["seed"])
    Xp = pca.fit_transform(Xs)
    log.info(
        "PCA %d comps explain %.1f%% variance", n_comp, 100 * pca.explained_variance_ratio_.sum()
    )

    feat_df = pd.DataFrame(Xp, columns=[f"f{i}" for i in range(n_comp)])
    feat_df["label"] = df["label"].values
    feat_df["crop_path"] = df["crop_path"].values
    feat_df.to_csv(out_csv, index=False)

    meta = {
        "pca_components": n_comp,
        "explained_variance_pct": float(100 * pca.explained_variance_ratio_.sum()),
        "n_samples": len(feat_df),
        "per_class": feat_df["label"].value_counts().to_dict(),
    }
    with open(ROOT / "data" / "features" / "features_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return feat_df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml")
    df = run_features(cfg)
    print(df["label"].value_counts())
    print("saved:", df.shape)
