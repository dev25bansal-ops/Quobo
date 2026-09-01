# Quobo — QUBO-based Quantum Feature Selection for Environmental Image Classification

IDP3 project: TACO trash dataset → MobileNetV2 features → PCA → QUBO selects 8 features → QSVM.
Four-arm ablation (all → same QSVM, 8-feature budget): **A** random · **B** PCA · **C** MI · **D** QUBO.
Plus pollution analytics: cleanliness score per zone + Pollution Severity Index (PSI).

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
configs/            YAML experiment configs (classes, feature budget, seeds, solver)
data/raw/           TACO dataset (downloaded by src/quobo/data_prep.py)
data/features/      MobileNetV2+PCA features as CSV (the QUBO/QSVM input)
src/quobo/          pipeline package
  data_prep.py      download TACO, crop annotations, build classification dataset
  features.py       MobileNetV2 embeddings -> PCA -> features.csv
  selection.py      arms A-D: random / PCA / MI / QUBO feature selection
  qsvm.py           ZZFeatureMap + FidelityStatevectorKernel + QSVC, RBF-SVM baseline
  run_experiment.py orchestrates the full A-D ablation
scripts/            entry points beyond the core pipeline
  confusion_analysis.py  per-class confusion matrices for the QUBO-8 arm
  pollution_dashboard.py zone cleanliness score + PSI + HTML dashboard
  fetch_missing.py       resumable TACO image fetcher (fallback URLs)
experiments/        per-run outputs (selected features, metrics)
results/            figures + tables + dashboard for the report
```

## Run

```bash
# full A-D ablation (config selectable; 6-class is the current paper run)
.venv/Scripts/python -m src.quobo.run_experiment configs/experiment_6class.yaml

# per-class confusion analysis (QUBO-8 arm)
.venv/Scripts/python -m scripts.confusion_analysis

# zone pollution dashboard (results/dashboard/dashboard.html)
.venv/Scripts/python -m scripts.pollution_dashboard

# data prep only (TACO download + crops) — already done for the paper run
.venv/Scripts/python -m src.quobo.data_prep configs/experiment_6class.yaml
```

## Status

- [x] Stage 1: data prep (TACO download + 1,836 crops, 6 classes)
- [x] Stage 2: features (MobileNetV2 → PCA-50 → CSV)
- [x] Stage 3: selection arms A-D
- [x] Stage 4: QSVM + RBF-SVM ablation (results/tables/summary_*.csv)
- [x] Stage 5: PSI / zone analytics (results/dashboard/dashboard.html)
