# Quobo — QUBO-based Quantum Feature Selection for Environmental Image Classification

IDP3 project: TACO trash dataset → MobileNetV2 features → leak-free per-split PCA → QUBO selects 8 features → QSVM.
Ablation arms (all → same classifier, 8-feature budget): **A** random · **B** PCA · **C** MI · **D** QUBO · **E** LASSO · **F** mRMR, plus a **Full-50** ceiling (no selection).
Plus pollution analytics: cleanliness score per zone + Pollution Severity Index (PSI, a project-defined relative index — unvalidated, not a health index).

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

**GNU Make** (optional, for the `make` shortcuts below) is a separate prerequisite —
it is *not* installed by `pip install -e .`:

```bash
scoop install make          # Windows (user-level, no admin)
sudo apt-get install make   # Debian/Ubuntu
brew install make           # macOS
```

Verify with `make --version` (expect GNU Make >= 4.0).

## Reproduce the paper's numbers

One command, from a fresh clone to the paper's tables:

```bash
.venv/Scripts/python -m scripts.reproduce_paper
```

It loads and validates the frozen config, reports environment drift against
`constraints.txt`, checks the on-disk dataset, runs (or reuses) the experiment,
and asserts the produced summary against the frozen golden baseline — with
the **same tolerance convention the statistical harness uses** (±1.5 sigma of
the baseline's own std, imported from `tests/test_golden_results.py` rather
than re-derived). Exit code 0 = all checks passed, 1 = a check failed.

```bash
# every check EXCEPT the experiment — cheap enough for CI:
.venv/Scripts/python -m scripts.reproduce_paper --dry-run

# reuse a completed run instead of running one:
.venv/Scripts/python -m scripts.reproduce_paper --run-id 20260920_161627
.venv/Scripts/python -m scripts.reproduce_paper --skip-run

# treat environment drift as a failure (a dev machine is not a paper machine):
.venv/Scripts/python -m scripts.reproduce_paper --strict
```

See `docs/reproducibility.md` (built into the Sphinx docs) for the full
description, including where the golden baseline comes from on a fresh clone.

## Makefile

`make` exposes one target per project verb. `make help` (the default) lists
them; `make verify` is the CI gate (lint + dry-run reproduction + leak audit).

| Target | What it does |
| --- | --- |
| `make test` | the full pytest suite |
| `make lint` | `ruff check` over `src/`, `scripts/`, `tests/` |
| `make fmt` / `make fmt-check` | `ruff format` in place / non-mutating check |
| `make docs` | Sphinx HTML build with `-W` (warnings are errors) |
| `make export` | ONNX export for edge inference |
| `make run` | the full ablation experiment |
| `make dashboard` | render the pollution dashboard |
| `make validate-data` | the offline data-contract validation |
| `make audit` | leakage audit of the newest complete run |
| `make verify` | CI gate: lint + dry-run reproduction + leak audit |
| `make reproduce` | reproduce the paper's numbers (see above) |
| `make build` | the continuous build (`PROFILE=quick\|full\|docs`) |
| `make clean` | remove caches and build artifacts |

Every recipe uses the venv interpreter explicitly
(`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere).

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
  reproduce_paper.py     one command from a fresh clone to the paper's numbers
  confusion_analysis.py  per-class confusion matrices (consumes saved predictions)
  pollution_dashboard.py zone cleanliness score + PSI + interactive HTML dashboard
  fetch_missing.py       resumable TACO image fetcher (fallback URLs)
  export_onnx.py         ONNX export for edge inference (backbone + feature + head)
  leak_audit.py          re-derive the leak-free claim from a run's artifacts
  prune_runs.py          retention policy for experiments/ (dry run by default)
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

# zone pollution dashboard — interactive Plotly charts (needs [dashboard]).
# Prediction mode requires a COMPLETED run with run_manifest.json + saved test_ids;
# legacy runs (before 2026-09-17) have neither and are refused.
.venv/Scripts/python -m scripts.pollution_dashboard
# explicit demo mode for old/legacy data — uses saved ground-truth labels, no model:
.venv/Scripts/python -m scripts.pollution_dashboard --mode demo-ground-truth

# ONNX export for edge inference (needs [edge]); --backbone re-exports MobileNetV2.
# results/onnx/export_manifest.json records size+sha256 for every shipped bundle
# file, including the backbone; without --backbone an existing
# backbone_mobilenet.onnx is reused and hashed (never regenerated).
.venv/Scripts/python -m scripts.export_onnx --config configs/experiment_6class.yaml [--backbone]

# data prep only (TACO download + crops) — already done for the paper run
.venv/Scripts/python -m src.quobo.data_prep configs/experiment_6class.yaml

# leakage audit of a completed run: re-derives the leak-free claim from the
# artifacts on disk. Read-only and offline — it never retrains and never
# replays a split, so it is safe in CI.
.venv/Scripts/python -m scripts.leak_audit
.venv/Scripts/python -m scripts.leak_audit --run-id 20260920_161627 --json

# retention policy for experiments/: REPORTS by default and deletes nothing
# unless --yes is passed. A run whose manifest says "running" is never deleted.
.venv/Scripts/python -m scripts.prune_runs --keep 5
```

## Docs & Tests

```bash
# build the Sphinx API reference (auto-generated from docstrings)
.venv/Scripts/python -m sphinx -b html docs docs/_build/html

# run the test suite (unit + property-based + integration + provenance).
.venv/Scripts/python -m pytest tests/ -q

# the authoritative test count (the number moves as the suite grows):
.venv/Scripts/python -m pytest --collect-only -q | tail -1

# continuous build (lint + tests + schema check; reports into vault/ notes)
pwsh scripts/continuous_build.ps1 -Profile quick|full|docs
```

### Test count

The suite is **~342 tests** (measured 2026-10-20). The number is not a
hand-maintained constant, because it changes every time a test is added. The
authoritative figure is printed by:

```bash
.venv/Scripts/python -m pytest --collect-only -q | tail -1
```

The machine-readable marker below is checked against a real collection by
`tests/test_reproduce.py` (opt-in: `QUOBO_CHECK_TEST_COUNT=1`, or
`make test-count`). If a test is added and the marker is not refreshed, that
check fails and says so.

<!-- test-count: 342 -->

## Knowledge base & automation

- `vault/` — Obsidian vault: architecture, module, research and build notes
  (start at `vault/00 Home.md`). `Build Status`/`Build Log` are auto-updated by
  `scripts/continuous_build.ps1`, which is scheduled 24/7 via CodeBuddy
  automations (hourly quick, daily full, weekly docs).

## Status

**As of 2026-09-20.** The artifacts below were produced by runs that predate the
provenance hardening. The full test suite (163 tests: unit, property-based,
integration, provenance, data-security, quantum-experiment) was re-run end to
end on the current working tree on 2026-09-20 and **passed green**
(163 passed, 11 benign warnings, ~3 min); `scripts/continuous_build.ps1` keeps
this verified via scheduled 24/7 builds.

- [x] Stage 1: data prep (TACO download + 1,836 crops, 6 classes)
- [x] Stage 2: features (MobileNetV2 → per-split PCA → cached embeddings)
- [x] Stage 3: selection arms A-F + Full-50 ceiling
- [x] Stage 4: QSVM + RBF-SVM ablation (results/tables/significance_*.csv, 25 repeats, Holm significance)
- [x] Stage 5: PSI / zone analytics (interactive dashboard) — PSI is a project-defined
      **relative, unvalidated index, not a health index**
- [x] Leak-free per-split PCA (no transductive leakage)
- [x] ONNX edge-inference export (100% parity vs sklearn on the exported head)
- [x] Full test-suite pass on the current tree (2026-09-20: 163 passed)

### Provenance requirements (2026-09-17)

The pollution dashboard's prediction mode refuses to guess provenance. A run is usable
only when:

- `experiments/<run_id>/run_manifest.json` exists with `status == "complete"`, a non-zero
  `reps_completed`, and a `{sha256, bytes}` inventory covering every artifact; and
- each `repN_<arm>.json` persists `test_ids`, `y_test`, and predictions of equal length —
  the saved crop IDs are the authority, so no train/test split is ever replayed.

Run directories created before 2026-09-17 carry neither a run manifest nor saved `test_ids`,
so they cannot be consumed in prediction mode. To reproduce the demo dashboard from those
runs, pass `--mode demo-ground-truth` explicitly (uses saved labels, no model).
