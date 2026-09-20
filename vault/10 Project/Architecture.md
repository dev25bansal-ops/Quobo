---
tags: [project, architecture]
up: "[[00 Home]]"
---

# Architecture & Data Flow

## Module dependency graph

```
configs/*.yaml ──> config.py (load_config / validate_config)
                         │
        ┌────────────────┼───────────────────────────────┐
        v                v                               v
  data_prep.py      features.py                    config_schema.py
  (Stage 1)         (Stage 2)                      (pydantic strict)
        │                │
        │                v
        │          selection.py (Stage 3, arms A–F/Z)
        │                │
        │                v
        │           qsvm.py (Stage 4)
        │                │
        └──> run_experiment.py  (_run_body: orchestration +
                    │            manifest + significance + MLflow)
      ┌─────────────┼──────────────┬──────────────────┐
      v             v              v                  v
 results/tables  experiments/  scripts/           scripts/pollution_
 results/figures <run_id>/      confusion_         dashboard.py (Stage 5)
 results/onnx    rep*_arm.json  analysis.py        └─> uncertainty.py
```

Side modules **not wired into the main pipeline**:
- [[data_validation]] — offline data-contract validator (`scripts/validate_data.py`)
- [[quantum_experiments]] — independent A2/A3/A4 quantum-kernel track

## Path anchor

`config.ROOT = Path(__file__).resolve().parents[2]` is the single repo-root anchor
used by every module for `data/`, `experiments/`, `results/` paths.

## Artifact layout

| Dir | Contents |
|---|---|
| `data/raw/` | TACO annotations + images + `annotations_provenance.json` |
| `data/processed/` | crops + `data_prep_stats.json` |
| `data/features/` | cached embeddings/PCA npz (fingerprint-keyed) |
| `experiments/<run_id>/` | `run_manifest.json` + `rep<N>_<arm>.json` per repeat |
| `results/tables/` | metrics + `significance_*.csv` (Holm-corrected) |
| `results/figures/` | report plots |
| `results/onnx/` | edge bundle + `export_manifest.json` (sha256 inventory) |

## Cross-cutting invariants

1. **Fail-fast config**: unknown keys raise (`config.KNOWN_KEYS`/`KNOWN_SUBKEYS`
   whitelist, manually synced with `config_schema.QuoboConfig` — keep both in step).
2. **No split replay**: saved `test_ids` in each rep JSON are the authority for
   downstream consumers ([[Provenance and Reproducibility]]).
3. **Fingerprint-keyed caches**: [[features]] caches embeddings by content
   fingerprint (`full-sha256-v1`); stale caches auto-invalidate.
   `data_validation._crop_inventory` **mirrors** the fingerprint formula — implicit
   coupling, change both together.
4. **Path safety**: all downloads/crops validated against traversal
   (`_validate_dataset_paths`, `is_relative_to` containment).

Related: [[Overview]] · [[Scripts]] · [[Issue Register]]
