---
tags: [scripts]
up: "[[00 Home]]"
---

# Scripts (`scripts/`)

Entry points beyond the core pipeline. All run as `python -m scripts.<name>`
with `.venv/Scripts/python` (Python 3.12).

| Script | Purpose | Needs |
|---|---|---|
| `confusion_analysis.py` | per-class confusion matrices for QUBO-8 arm; **consumes saved predictions** (never re-splits) | completed run |
| `pollution_dashboard.py` | Stage 5: zone cleanliness + PSI + interactive Plotly HTML | `[dashboard]`; prediction mode needs valid manifest |
| `fetch_missing.py` | resumable TACO image fetcher with fallback URLs | network |
| `export_onnx.py` | ONNX edge bundle (backbone+feature+head), writes `export_manifest.json` | `[edge]`; `--backbone` to re-export |
| `validate_data.py` | CLI for [[data_validation]] contract checks | — |
| `qpu_run.py` | real D-Wave QPU EXP-D run | `DWAVE_API_TOKEN` (skips green without) |
| `ibm_kernel.py` | IBM Quantum kernel execution | IBM token |
| `kernel_ablation.py` | drives [[quantum_experiments]]`run_ablation` grid | qiskit |
| `train_quantum_kernel.py` | trains feature-map alignment (A2/A3) | qiskit |
| `continuous_build.ps1` | **the 24/7 build orchestrator** — see [[Continuous Build]] | PowerShell |

## Dashboard modes

- **prediction** (default): requires `run_manifest.json` `status=="complete"` +
  saved `test_ids`; raises `ProvenanceError` otherwise (PRV-003).
- **demo**: `--mode demo-ground-truth` — uses saved labels, no model; the only
  way to render legacy (pre-2026-09-17) runs.

## PSI disclaimer

PSI is a **project-defined relative index, unvalidated, not a health index**;
bands/scale borrowed from air-quality indices for readability only (PSI-002).

Related: [[Runbook]] · [[Test Suite]]
