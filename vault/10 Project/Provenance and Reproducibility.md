---
tags: [project, provenance]
up: "[[00 Home]]"
---

# Provenance & Reproducibility

Hardening pass of 2026-09-17 (Q04/Q05, PRV-001..004, ONNX-001). Enforced rules:

## Run manifests

- Every experiment run writes `experiments/<run_id>/run_manifest.json` with
  lifecycle `running → complete | failed`, `reps_completed`, config digest
  (`_config_digest`), and a `{sha256, bytes}` inventory of every artifact
  (`_artifact_inventory`, atomic writes via `_write_json_atomic`).
- **Consumers must require `status == "complete"`** and a non-zero `reps_completed`.

## Prediction persistence

- Each `rep<N>_<arm>.json` persists `test_ids`, `y_test`, and aligned predictions
  of equal length. Saved crop IDs are the authority — **no train/test split is
  ever replayed** by downstream tools.
- [[Overview|Dashboard]] prediction mode raises `ProvenanceError` on
  missing/corrupt/tampered/incomplete runs; no silent fallback to ground truth.
  Ground-truth demo requires explicit `--mode demo-ground-truth`.

## Data provenance

- `data/raw/annotations_provenance.json`: sha256 `ac1ec605…`, n=1500 images
  pinned at download time (OPEN-6).
- Feature caches keyed by **content fingerprint** (`full-sha256-v1`: per-file
  size + streaming SHA-256) — see [[features]].

## ONNX bundle

- `results/onnx/export_manifest.json` is a `{sha256, bytes}` inventory covering
  `features_1280_50.onnx`, `head_svm_8.onnx`, `backbone_mobilenet.onnx`,
  `inference.py`. Existing backbone is reused/hashed, never silently regenerated.

## Reproducibility

- Python 3.12 + `constraints.txt` (exact pins that produced paper numbers).
- Same seed → same QUBO selection (verified in the leak-free rerun).
- Statistical protocol: 25 group-aware repeats, paired significance with Holm
  correction ([[Key Results]]).

## Known limitation

- **PRV-004**: legacy run dirs (pre 2026-09-17) have no manifest/test_ids and are
  refused in prediction mode. Fix = re-run, or use demo mode.

Related: [[run_experiment]] · [[data_validation]] · [[Issue Register]]
