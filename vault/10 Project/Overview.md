---
tags: [project, overview]
up: "[[00 Home]]"
---

# Overview

**Quobo** (v1.0.0, `quobo` package, src-layout) is an IDP3 research project:
TACO trash images → MobileNetV2 embeddings → leak-free per-split PCA (50 dims) →
**QUBO selects 8 features** → QSVM classification, with a 7-arm selection ablation
and a pollution-analytics dashboard.

## Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| 1 | [[data_prep]] | Download TACO (1,500 imgs → 1,836 crops, 6 classes), bbox crops with 10% margin |
| 2 | [[features]] | MobileNetV2 embeddings (checkpointed) → per-split MinMax+PCA → cached npz |
| 3 | [[selection]] | Arms A–F: random / PCA-order / MI / **QUBO** / LASSO / mRMR, + Full-50 ceiling |
| 4 | [[qsvm]] | ZZFeatureMap + FidelityStatevectorKernel + QSVC; RBF-SVM / linear / tuned baselines |
| 5 | `scripts/pollution_dashboard.py` | Zone cleanliness score + PSI (project-defined relative index) + Plotly HTML |

Orchestration: [[run_experiment]] (configs in `configs/*.yaml`, e.g. `experiment_6class.yaml`).

## Ablation arms (8-feature budget, same classifier)

A random · B PCA · C MI · **D QUBO** · E LASSO · F mRMR · Z Full-50 (no selection).
Headline numbers in [[Key Results]].

## Key properties

- **Leak-free**: per-split scaler+PCA fit (no transductive leakage), GroupShuffleSplit by source photo.
- **Provenance**: every run writes `run_manifest.json` (`running→complete/failed`) with `{sha256, bytes}` artifact inventory; consumers refuse incomplete runs — see [[Provenance and Reproducibility]].
- **Reproducible**: Python 3.12 pinned via `constraints.txt` (153 pins); same seed → same selection (verified).
- **Edge export**: ONNX bundle with 100% parity vs sklearn (`scripts/export_onnx.py`).

## Current state (2026-09-20)

- Stages 1–5 done; paper on leak-free v2 numbers.
- Working tree carries uncommitted provenance hardening + new quantum-kernel track ([[quantum_experiments]]).
- Open items are externally blocked — see [[Issue Register]].

Related: [[Architecture]] · [[Runbook]] · [[Scripts]]
