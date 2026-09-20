---
tags: [module]
path: src/quobo/features.py
up: "[[00 Home]]"
---

# features — Stage 2

MobileNetV2 embeddings → **leak-free per-split PCA (50 dims)** → cached npz.

## Caching & fingerprints

- `crops_fingerprint` — filename/order/count hash (sha256, first 16 hex) → `files_hash`.
- `crops_content_fingerprint` — Q07: full per-file size + streaming SHA-256, `hash_version="full-sha256-v1"`, chunk 1 MiB.
- `_features_config_identity(cfg)` — config keys that invalidate the cache (backbone, pca_components…).
- Cache auto-invalidation is live-tested (CCH-001 fix): stale fingerprint → recompute.
- ⚠️ `data_validation._crop_inventory` **mirrors** the fingerprint formula — implicit coupling; change both together.

## Leak-free protocol (OPEN-5 fix)

Scaler + PCA are fit **separately per split** from cached raw embeddings — no
transductive leakage. This changed the paper numbers (v1 → v2); full 25-repeat
rerun took 10.7 min. See [[Key Results]].

## Embedding

- Backbone: `tf.keras.applications.MobileNetV2`, weights cached; checkpointed
  per-batch progress so interrupted runs resume.
- Output: raw 1280-d embeddings + 50-d PCA features per split, npz under
  `data/features/`.

Consumers: [[selection]] (operates on the 50-d PCA space), [[qsvm]].

Related: [[Provenance and Reproducibility]] · [[data_validation]]
