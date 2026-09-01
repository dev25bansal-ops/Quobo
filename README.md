# Quobo — QUBO-based Quantum Feature Selection for Environmental Image Classification

IDP3 project: TACO trash dataset → MobileNetV2 features → PCA → QUBO selects 8 features → QSVM.
Four-arm ablation (all → same QSVM, 8-feature budget): **A** random · **B** PCA · **C** MI · **D** QUBO.
Plus pollution analytics: cleanliness score per zone + Pollution Severity Index (PSI) mapped in QGIS.

## Setup

Requires Python 3.12 (pinned: qiskit 2.x, qiskit-machine-learning 0.9.x, TF 2.19).

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Optional: D-Wave Leap token for real-QPU EXP D runs (`set DWAVE_API_TOKEN=...`). Everything
else runs locally free — QUBO on simulated annealing, QSVM on statevector simulator.

## Layout

```
configs/            YAML experiment config (classes, feature budget, seeds, solver)
data/raw/           TACO dataset (downloaded by src/quobo/data_prep.py)
data/features/      MobileNetV2+PCA features as CSV (the QUBO/QSVM input)
src/quobo/          pipeline package
  data_prep.py      download TACO, crop annotations, build classification dataset
  features.py       MobileNetV2 embeddings -> PCA -> features.csv
  selection.py      arms A-D: random / PCA / MI / QUBO feature selection
  qsvm.py           ZZFeatureMap + FidelityQuantumKernel + QSVC, RBF-SVM baseline
  run_experiment.py orchestrates the full A-D ablation
  psi.py            zone-level cleanliness score + Pollution Severity Index
experiments/        per-run outputs (selected features, metrics, kernel caches)
results/            figures + tables for the report
```

## Run

```bash
.venv/Scripts/python -m src.quobo.run_experiment --config configs/experiment.yaml
```

## Status

- [ ] Stage 1: data prep (TACO download + crops)
- [ ] Stage 2: features (MobileNetV2 → PCA-50 → CSV)
- [ ] Stage 3: selection arms A-D
- [ ] Stage 4: QSVM + RBF-SVM ablation
- [ ] Stage 5: PSI / zone analytics
