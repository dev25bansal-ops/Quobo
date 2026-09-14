# Quobo — QUBO-based Quantum Feature Selection for Environmental Image Classification

IDP3 project: TACO trash dataset → MobileNetV2 features → leak-free per-split PCA → QUBO selects 8 features → QSVM.
Ablation arms (all → same classifier, 8-feature budget): **A** random · **B** PCA · **C** MI · **D** QUBO · **E** LASSO · **F** mRMR, plus a **Full-50** ceiling (no selection).
Plus pollution analytics: cleanliness score per zone + Pollution Severity Index (PSI).

## Setup

Requires Python 3.12 (pinned in `constraints.txt` — the exact versions that produced the paper numbers).

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt          # core pipeline
# optional extras (all opt-in, degrade gracefully if absent):
.venv/Scripts/python -m pip install ".[edge]"       # ONNX export (onnxruntime/skl2onnx/tf2onnx)
.venv/Scripts/python -m pip install ".[dashboard]"  # interactive Plotly dashboard
.venv/Scripts/python -m pip install ".[track]"      # MLflow experiment tracking
```

Optional: D-Wave Leap token for real-QPU EXP D runs (`set DWAVE_API_TOKEN=...`). Everything
else runs locally free — QUBO on simulated annealing, QSVM on statevector simulator.

## Layout

```
configs/            YAML experiment configs (classes, feature budget, seeds, solver)
data/raw/           TACO dataset (downloaded by src/quobo/data_prep.py)
data/features/      MobileNetV2 embeddings + PCA features (the QUBO/QSVM input)
src/quobo/          pipeline package
  config.py         config loading + schema validation (fail-fast on bad config)
  config_schema.py  Pydantic schema for the experiment config
  data_prep.py      download TACO, crop annotations, build classification dataset
  features.py       MobileNetV2 embeddings (checkpointed) -> per-split PCA -> features
  selection.py      arms A-F: random / PCA / MI / QUBO / LASSO / mRMR
  qsvm.py           ZZFeatureMap + FidelityStatevectorKernel + QSVC, RBF-SVM baseline
  run_experiment.py orchestrates the full ablation + significance + MLflow
scripts/            entry points beyond the core pipeline
  confusion_analysis.py  per-class confusion matrices (consumes saved predictions)
  pollution_dashboard.py zone cleanliness score + PSI + interactive HTML dashboard
  fetch_missing.py       resumable TACO image fetcher (fallback URLs)
  export_onnx.py         ONNX export for edge inference (backbone + feature + head)
docs/               Sphinx API reference (auto-generated from docstrings)
experiments/        per-run outputs (selected features, metrics)
results/            figures + tables + dashboard + onnx for the report
```

## Run

```bash
# full ablation (config selectable; 6-class is the current paper run).
# --track logs the run to MLflow (needs [track]); off by default.
.venv/Scripts/python -m src.quobo.run_experiment configs/experiment_6class.yaml [--track]

# per-class confusion analysis (QUBO-8 arm; consumes saved predictions)
.venv/Scripts/python -m scripts.confusion_analysis

# zone pollution dashboard — interactive Plotly charts (needs [dashboard])
.venv/Scripts/python -m scripts.pollution_dashboard

# ONNX export for edge inference (needs [edge]); --backbone adds MobileNetV2
.venv/Scripts/python -m scripts.export_onnx --config configs/experiment_6class.yaml [--backbone]

# data prep only (TACO download + crops) — already done for the paper run
.venv/Scripts/python -m src.quobo.data_prep configs/experiment_6class.yaml
```

## Docs & Tests

```bash
# build the Sphinx API reference (auto-generated from docstrings)
.venv/Scripts/python -m sphinx -b html docs docs/_build/html

# run the test suite (unit + property-based + integration; 32 tests)
.venv/Scripts/python -m pytest tests/ -q
```

## Status

- [x] Stage 1: data prep (TACO download + 1,836 crops, 6 classes)
- [x] Stage 2: features (MobileNetV2 → per-split PCA → cached embeddings)
- [x] Stage 3: selection arms A-F + Full-50 ceiling
- [x] Stage 4: QSVM + RBF-SVM ablation (results/tables/summary_*.csv, 25 repeats, Holm significance)
- [x] Stage 5: PSI / zone analytics (interactive dashboard)
- [x] Leak-free per-split PCA (no transductive leakage)
- [x] ONNX edge-inference export (100% parity vs sklearn)
- [x] Tests (32), ruff lint, Sphinx docs, MLflow tracking
