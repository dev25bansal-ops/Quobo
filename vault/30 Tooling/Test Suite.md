---
tags: [tests]
up: "[[00 Home]]"
---

# Test Suite (`tests/`)

~107 test functions across 9 files. Run: `.venv/Scripts/python -m pytest tests/ -q`.

| File | Covers | Kind |
|---|---|---|
| `test_pipeline.py` | core stages: config, data_prep, features, selection, qsvm, manifest, ONNX parity | unit + integration |
| `test_integration.py` | end-to-end pipeline smoke | integration (marker) |
| `test_properties.py` | leak-free PCA shape/statistical invariants | property (hypothesis) |
| `test_data_security.py` | path traversal, hostile JSON, download caps in [[data_prep]] | unit/security |
| `test_data_validation.py` | [[data_validation]] contract checks | unit |
| `test_hardening.py` | prior audit fixes (MinMaxScaler leak, PSI weights, bbox clamp, cache invalidation…) | regression |
| `test_provenance.py` | manifest lifecycle, `test_ids` alignment, dashboard refusal | unit |
| `test_psi_uncertainty.py` | PSI math + [[uncertainty]] Wilson/vote | unit |
| `test_quantum_experiments.py` | [[quantum_experiments]] splits/kernels/ablation | unit |

## Baseline (2026-09-20)

**163 passed, 11 warnings in 171.37s** — full suite is GREEN on the current
working tree (this supersedes the README's "full green run not claimed").
Warnings are benign (sklearn PCA divide RuntimeWarning, SVC probA_/probB_ deprecation).

## Offline policy

- **No network required**: tests use tiny synthetic datasets/fixtures; the
  integration test builds a real TF model architecture but downloads **no
  weights** (marker `integration`, skipped by pre-commit, run in CI).
- `hypothesis` property tests for numeric invariants.

## Status

- 2026-09-10: 16/16 green (older suite).
- 2026-09-17: ~107 tests present, full green run not claimed on the working tree.
- **2026-09-20: 163 passed, full suite GREEN** (see Baseline above).
- Live status: see [[Build Status]] (updated by [[Continuous Build]]).

## Markers

```toml
# pyproject.toml
markers = ["integration: end-to-end pipeline test (real TF model, no weight download)"]
```

Fast loop: `pytest tests/ -m "not integration"` · Full: `pytest tests/`.
