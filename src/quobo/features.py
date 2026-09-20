"""Stage 2: feature extraction — MobileNetV2 embeddings -> PCA -> features.csv.

Each crop is resized to 224x224, passed through MobileNetV2 (ImageNet weights,
global-average-pooled -> 1280-d), then PCA compresses to N candidate features.
Output: data/features/features.csv with columns f0..f{N-1}, label, crop_path.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import ROOT

log = logging.getLogger(__name__)

# Bound memory, not coverage: hash every byte, including large-image tails.
FINGERPRINT_CHUNK_BYTES = 1 << 20


def crops_fingerprint(crops_dir: Path, classes: list[str]) -> dict:
    """Name/order/count identity of the crop set a features file was computed
    from: per-class counts plus a hash over the sorted file list.

    Deliberately content-blind and dependency-light; ``data_validation`` mirrors
    this exact formula. Use :func:`crops_content_fingerprint` for cache identity."""
    per_class, names = {}, []
    for cls in classes:
        files = (
            sorted(p.name for p in (crops_dir / cls).glob("*.jpg"))
            if (crops_dir / cls).is_dir()
            else []
        )
        per_class[cls] = len(files)
        names.extend(f"{cls}/{n}" for n in files)
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    return {"per_class": per_class, "n_crops": len(names), "files_hash": digest}


def _file_content_digest(path: Path) -> tuple[int, str]:
    """Return full-file size and SHA-256 using bounded-memory reads."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(FINGERPRINT_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def crops_content_fingerprint(crops_dir: Path, classes: list[str]) -> dict:
    """Q07 content+size aware identity of the crop set (superset of
    :func:`crops_fingerprint`).

    Hashes the ordered class/file names together with each file's full size and
    full streamed content. Embedding/feature caches and resume
    checkpoints key on this, so byte-level corrections, dataset swaps and
    reordering force a recompute instead of silently reusing stale outputs."""
    per_class, entries = {}, []
    for cls in classes:
        cls_dir = crops_dir / cls
        files = sorted(cls_dir.glob("*.jpg")) if cls_dir.is_dir() else []
        per_class[cls] = len(files)
        for p in files:
            size, content = _file_content_digest(p)
            entries.append(f"{cls}/{p.name}:{size}:{content}")
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]
    return {
        "per_class": per_class,
        "n_crops": len(entries),
        "content_hash": digest,
        "hash_version": "full-sha256-v1",
    }


def _features_config_identity(cfg: dict) -> dict:
    """Config fields that change the produced features (Q07)."""
    features = cfg.get("features", {})
    return {
        "backbone": features.get("backbone", "MobileNetV2"),
        "pca_components": features.get("pca_components"),
    }


def load_model() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )
    return base


def _atomic_savez(path: Path, **arrays) -> None:
    """Publish a complete archive without overwriting a valid prior checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def extract_embeddings(
    model,
    crops_dir: Path,
    classes: list[str],
    checkpoint_dir: Path | None = None,
    checkpoint_tag: str | None = None,
    retain_checkpoint: bool = False,
) -> pd.DataFrame:
    """Extract 1280-d MobileNetV2 embeddings. If checkpoint_dir is given,
    progress is saved per batch to <checkpoint_dir>/<checkpoint_tag>.partial.npz
    and a crashed run resumes from the last completed batch (skipping
    already-embedded paths) instead of restarting. Checkpoints must match the
    crop fingerprint, ordered paths, output width and completion-mask shape.
    Legacy checkpoints without identity metadata are recomputed. Set
    retain_checkpoint=True when the caller must publish a final cache before
    removing recovery state. checkpoint_dir=None keeps the one-shot behavior."""
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
    n = len(paths)
    embs = np.zeros((n, emb_dim), dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    # resume from a prior checkpoint if present (crash recovery)
    ckpt_path = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{checkpoint_tag or 'default'}.partial.npz"
        identity = json.dumps(crops_content_fingerprint(crops_dir, classes), sort_keys=True)
        if ckpt_path.exists():
            try:
                with np.load(ckpt_path, allow_pickle=False) as z:
                    valid = (
                        str(z["fingerprint"].item()) == identity
                        and list(z["paths"]) == paths
                        and z["done"].shape == (n,)
                        and z["done"].dtype == np.dtype(bool)
                        and z["embs"].shape == (n, emb_dim)
                        and np.isfinite(z["embs"][z["done"]]).all()
                    )
                    if valid:
                        done = z["done"].copy()
                        embs[done] = z["embs"][done]
                        log.info("resuming embeddings: %d/%d already done", int(done.sum()), n)
                    else:
                        log.warning("checkpoint identity or shape mismatch; restarting extraction")
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                log.warning("checkpoint unreadable or legacy format; restarting extraction")

    bs = 64
    for i in tqdm(range(0, n, bs), desc="embeddings"):
        chunk = paths[i : i + bs]
        chunk_idx = np.arange(i, min(i + bs, n))  # clip to n for the last partial batch
        # skip any rows already embedded (resume path)
        todo_mask = ~done[chunk_idx]
        if not todo_mask.any():
            continue
        todo_pos = np.where(todo_mask)[0]
        todo_paths = [chunk[j] for j in todo_pos]
        imgs = np.stack([tf.keras.utils.load_img(p, target_size=(224, 224)) for p in todo_paths])
        imgs = tf.keras.applications.mobilenet_v2.preprocess_input(imgs)
        embs[chunk_idx[todo_pos], :] = model.predict(imgs, verbose=0)
        done[chunk_idx[todo_pos]] = True
        if ckpt_path is not None:
            _atomic_savez(
                ckpt_path,
                paths=np.array(paths),
                done=done,
                embs=embs,
                fingerprint=np.array(identity),
            )

    if ckpt_path is not None:
        _atomic_savez(
            ckpt_path, paths=np.array(paths), done=done, embs=embs, fingerprint=np.array(identity)
        )
        if not retain_checkpoint:
            ckpt_path.unlink(missing_ok=True)
    return pd.DataFrame(embs, columns=[f"e{i}" for i in range(emb_dim)]).assign(
        label=labels, crop_path=paths
    )


def run_raw_embeddings(cfg: dict) -> pd.DataFrame:
    """Cache the raw MobileNetV2 embeddings (1280-d) + labels + crop paths.

    The leak-free protocol (OPEN-5) refits scaler+PCA inside each train/test
    split from this cache; the legacy transductive PCA CSV is still produced
    by run_features for backward compatibility."""
    crops_dir = ROOT / (cfg["data"].get("crops_dir_override") or "data/processed/crops")
    classes = cfg["data"]["classes"]
    tag = "raw_" + "_".join(sorted(classes))
    out_npz = ROOT / "data" / "features" / f"{tag}.npz"
    meta_path = ROOT / "data" / "features" / f"{tag}_meta.json"
    ckpt_dir = ROOT / "data" / "features" / "_checkpoints"
    fp = crops_fingerprint(crops_dir, classes)
    cfp = crops_content_fingerprint(crops_dir, classes)

    if out_npz.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("crops_fingerprint") == fp and meta.get("crops_content_fingerprint") == cfp:
                log.info("raw embeddings cache hit: %s", out_npz.name)
                return load_raw_embeddings(out_npz)
            log.warning("raw cache identity mismatch — recomputing embeddings")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            log.warning("raw cache meta missing — recomputing embeddings")
    # The extractor validates checkpoint identity; a cache miss alone must not
    # discard resumable work. Keep it until the final cache and metadata exist.
    model = load_model()
    df = extract_embeddings(
        model,
        crops_dir,
        classes,
        checkpoint_dir=ckpt_dir,
        checkpoint_tag=tag,
        retain_checkpoint=True,
    )
    emb_cols = [c for c in df.columns if c.startswith("e")]
    _atomic_savez(
        out_npz,
        embeddings=df[emb_cols].values.astype(np.float32),
        labels=np.asarray(df["label"].values, dtype="U"),  # unicode, not object
        crop_paths=np.asarray(df["crop_path"].values, dtype="U"),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crops_fingerprint": fp,
                "crops_content_fingerprint": cfp,
                "features_config": _features_config_identity(
                    cfg
                ),  # informational (raw is PCA-free)
                "n_samples": len(df),
            },
            f,
            indent=2,
        )
    (ckpt_dir / f"{tag}.partial.npz").unlink(missing_ok=True)
    log.info("raw embeddings cached: %s (%d x %d)", out_npz.name, len(df), len(emb_cols))
    return load_raw_embeddings(out_npz)


def load_raw_embeddings(out_npz: Path) -> pd.DataFrame:
    with np.load(out_npz, allow_pickle=False) as z:
        n_comp = z["embeddings"].shape[1]
        return pd.DataFrame(z["embeddings"], columns=[f"e{i}" for i in range(n_comp)]).assign(
            label=z["labels"], crop_path=z["crop_paths"]
        )


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
    crops_dir = ROOT / (cfg["data"].get("crops_dir_override") or "data/processed/crops")
    classes = cfg["data"]["classes"]
    tag = "features_" + "_".join(sorted(classes))
    out_csv = ROOT / "data" / "features" / f"{tag}.csv"
    meta_path = ROOT / "data" / "features" / f"{tag}_meta.json"
    fp = crops_fingerprint(crops_dir, classes)
    cfp = crops_content_fingerprint(crops_dir, classes)
    cfg_id = _features_config_identity(cfg)

    if out_csv.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            cached_fp = meta["crops_fingerprint"]
            cached_cfp = meta.get("crops_content_fingerprint")
            cached_cfg = meta.get("features_config")
            if cached_fp == fp and cached_cfp == cfp and cached_cfg == cfg_id:
                log.info("features exist and match current crops/config: %s", out_csv)
                return pd.read_csv(out_csv)
            log.warning(
                "cached features DO NOT match crops/config on disk "
                "(content %s vs %s; config %s vs %s) — recomputing",
                (cached_cfp or {}).get("content_hash"),
                cfp["content_hash"],
                cached_cfg,
                cfg_id,
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
        "crops_content_fingerprint": cfp,
        "features_config": cfg_id,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return feat_df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment_6class.yaml")
    df = run_features(cfg)
    print(df["label"].value_counts())
    print("saved:", df.shape)
