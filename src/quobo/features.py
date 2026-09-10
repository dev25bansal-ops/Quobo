"""Stage 2: feature extraction — MobileNetV2 embeddings -> PCA -> features.csv.

Each crop is resized to 224x224, passed through MobileNetV2 (ImageNet weights,
global-average-pooled -> 1280-d), then PCA compresses to N candidate features.
Output: data/features/features.csv with columns f0..f{N-1}, label, crop_path.
"""
import hashlib
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


def crops_fingerprint(crops_dir: Path, classes: list[str]) -> dict:
    """Identity of the crop set a features file was computed from: per-class
    counts plus a hash over the sorted file list. Stored in features_meta.json
    and validated before any cached CSV is reused."""
    per_class, names = {}, []
    for cls in classes:
        files = sorted(p.name for p in (crops_dir / cls).glob("*.jpg")) if (crops_dir / cls).is_dir() else []
        per_class[cls] = len(files)
        names.extend(f"{cls}/{n}" for n in files)
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    return {"per_class": per_class, "n_crops": len(names), "files_hash": digest}


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

    # probe the model's output dimension once (MobileNetV2 -> 1280; test
    # backbones may differ) so any embedding width is supported
    probe = np.zeros((1, 224, 224, 3), dtype=np.float32)
    emb_dim = int(np.asarray(model.predict(probe, verbose=0)).shape[1])
    embs = np.zeros((len(paths), emb_dim), dtype=np.float32)
    bs = 64
    for i in tqdm(range(0, len(paths), bs), desc="embeddings"):
        chunk = paths[i : i + bs]
        imgs = np.stack(
            [tf.keras.utils.load_img(p, target_size=(224, 224)) for p in chunk]
        )
        imgs = tf.keras.applications.mobilenet_v2.preprocess_input(imgs)
        embs[i : i + bs] = model.predict(imgs, verbose=0)
    return pd.DataFrame(embs, columns=[f"e{i}" for i in range(emb_dim)]).assign(
        label=labels, crop_path=paths
    )


def run_raw_embeddings(cfg: dict) -> pd.DataFrame:
    """Cache the raw MobileNetV2 embeddings (1280-d) + labels + crop paths.

    The leak-free protocol (OPEN-5) refits scaler+PCA inside each train/test
    split from this cache; the legacy transductive PCA CSV is still produced
    by run_features for backward compatibility."""
    crops_dir = ROOT / cfg["data"].get("crops_dir_override", "data/processed/crops")
    classes = cfg["data"]["classes"]
    tag = "raw_" + "_".join(sorted(classes))
    out_npz = ROOT / "data" / "features" / f"{tag}.npz"
    meta_path = ROOT / "data" / "features" / f"{tag}_meta.json"
    fp = crops_fingerprint(crops_dir, classes)

    if out_npz.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                cached_fp = json.load(f)["crops_fingerprint"]
            if cached_fp == fp:
                log.info("raw embeddings cache hit: %s", out_npz.name)
                return load_raw_embeddings(out_npz)
            log.warning("raw cache fingerprint mismatch — recomputing embeddings")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            log.warning("raw cache meta missing — recomputing embeddings")
        out_npz.unlink(missing_ok=True)

    model = load_model()
    df = extract_embeddings(model, crops_dir, classes)
    emb_cols = [c for c in df.columns if c.startswith("e")]
    np.savez_compressed(
        out_npz,
        embeddings=df[emb_cols].values.astype(np.float32),
        labels=np.asarray(df["label"].values, dtype="U"),  # unicode, not object
        crop_paths=np.asarray(df["crop_path"].values, dtype="U"),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"crops_fingerprint": fp, "n_samples": len(df)}, f, indent=2)
    log.info("raw embeddings cached: %s (%d x %d)", out_npz.name, len(df), len(emb_cols))
    return load_raw_embeddings(out_npz)


def load_raw_embeddings(out_npz: Path) -> pd.DataFrame:
    with np.load(out_npz, allow_pickle=False) as z:
        n_comp = z["embeddings"].shape[1]
        return pd.DataFrame(
            z["embeddings"], columns=[f"e{i}" for i in range(n_comp)]
        ).assign(label=z["labels"], crop_path=z["crop_paths"])


def fit_pca_on_train(Xtr_raw: np.ndarray, Xall_raw: np.ndarray, n_comp: int, seed: int):
    """Fit StandardScaler+PCA on TRAIN rows only, transform all rows.
    Returns (Xtr_pca, Xall_pca, explained_var_pct). This is the leak-free
    replacement for the transductive global fit."""
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr_s = scaler.transform(Xtr_raw)
    Xall_s = scaler.transform(Xall_raw)
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xtr_s)
    Xtr_p = pca.transform(Xtr_s)
    Xall_p = pca.transform(Xall_s)
    return Xtr_p, Xall_p, float(100 * pca.explained_variance_ratio_.sum())


def run_features(cfg: dict) -> pd.DataFrame:
    """Legacy cached PCA features (transductive fit — kept for dashboard/
    confusion consumers). The experiment path uses run_raw_embeddings +
    fit_pca_on_train instead (leak-free)."""
    crops_dir = ROOT / cfg["data"].get("crops_dir_override", "data/processed/crops")
    classes = cfg["data"]["classes"]
    tag = "features_" + "_".join(sorted(classes))
    out_csv = ROOT / "data" / "features" / f"{tag}.csv"
    meta_path = ROOT / "data" / "features" / f"{tag}_meta.json"
    fp = crops_fingerprint(crops_dir, classes)

    if out_csv.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                cached_fp = json.load(f)["crops_fingerprint"]
            if cached_fp == fp:
                log.info("features exist and match current crops: %s", out_csv)
                return pd.read_csv(out_csv)
            log.warning(
                "cached features DO NOT match crops on disk (hash %s vs %s) — recomputing",
                cached_fp.get("files_hash"), fp["files_hash"],
            )
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            log.warning("features meta missing/unreadable — recomputing to be safe")
        out_csv.unlink(missing_ok=True)

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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out_csv, index=False)

    meta = {
        "pca_components": n_comp,
        "explained_variance_pct": float(100 * pca.explained_variance_ratio_.sum()),
        "n_samples": len(feat_df),
        "per_class": feat_df["label"].value_counts().to_dict(),
        "crops_fingerprint": fp,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
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
