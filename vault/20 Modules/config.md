---
tags: [module]
path: src/quobo/config.py, src/quobo/config_schema.py
up: "[[00 Home]]"
---

# config & config_schema

Fail-fast configuration: load YAML → whitelist check → pydantic validation.

## `config.py` (102 lines)

- `ROOT = Path(__file__).resolve().parents[2]` — repo-root anchor for all modules.
- `KNOWN_KEYS` — 7 top-level sections: `seed, data, features, selection, qubo, qsvm, experiment`.
- `KNOWN_SUBKEYS` — per-section whitelist; **must be kept manually in sync with `config_schema.QuoboConfig`**.
- `load_config(path)` — raises `ValueError` on unknown keys (Q17).
- `validate_config(cfg)` — lazy-imports pydantic model, wraps errors as `ValueError("invalid experiment config: …")`.
- `ensure_dirs(cfg)` — creates `data/raw|processed|features`, `experiments`, `results/figures|tables`.

## `config_schema.py` (138 lines) — pydantic v2, `extra="forbid"`

| Section | Key constraints |
|---|---|
| `DataConfig` | classes ≥2, min_images_per_class ≥10, `crops_dir_override` (TrashNet replication) |
| `FeaturesConfig` | backbone `Literal["MobileNetV2"]`, pca_components 10..200 |
| `SelectionConfig` | k 2..50 |
| `QUBOConfig` | mi_bins 4..32, sa_sweeps ≥1000, sa_repeats 5..50, parallel |
| `QSVMConfig` | reps 1..5, entanglement linear/full, max_train_samples ≥50, test_fraction 0.1..0.5, tune_rbf, max_qsvm_features ≤30 (default 20) |
| `ExperimentConfig` | n_repeats 5..100, arms (7 literals), early_stop, track_mlflow, baselines |

Cross-field validator (Q19): `selection.k <= features.pca_components`.

## Gotchas

- `qsvm.max_memory_mb` is accepted by schema but has **no consumer in code**.
- pydantic is now declared in `pyproject.toml` (was a gap until 2026-09-20).

Related: [[Architecture]] · [[run_experiment]]
