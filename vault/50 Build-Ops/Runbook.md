---
tags: [ops, runbook]
up: "[[00 Home]]"
---

# Runbook

## Environment

- Python **3.12** venv at `.venv/` (currently 3.12.8); exact pins in
  `constraints.txt` (153 incl. pydantic, added 2026-09-20).
- Extras: `[dev]` (pytest/ruff/hypothesis/sphinx) installed; `[dashboard]`,
  `[edge]`, `[track]` opt-in.
- Rebuild env: `py -3.12 -m venv .venv` →
  `.venv/Scripts/python -m pip install -r requirements.txt -c constraints.txt` →
  `.venv/Scripts/python -m pip install -e ".[dev]"`.

## Commit-time gate (pre-commit)

Installed 2026-09-20 into `.git/hooks/pre-commit` (config:
`.pre-commit-config.yaml` — ruff --fix + ruff-format + pytest
`-m "not integration"`). Hooks resolve `python` from PATH, so commit with the
venv active (or the IDE terminal default). Re-install after venv rebuild:
`.venv/Scripts/python -m pip install pre-commit && pre-commit install`.
⚠️ ruff-format rewrites staged files — review `git diff` after blocked commits.

## Daily operations

```powershell
# one build (updates this vault automatically)
pwsh scripts/continuous_build.ps1 -Profile quick   # ~2 min, no integration
pwsh scripts/continuous_build.ps1 -Profile full    # ~3 min, incl. integration
pwsh scripts/continuous_build.ps1 -Profile docs    # + Sphinx

# raw commands
.venv/Scripts/python -m ruff check src scripts tests
.venv/Scripts/python -m pytest tests/ -q
.venv/Scripts/python -m sphinx -b html docs docs/_build/html
```

## 24/7 automation (CodeBuddy scheduled tasks)

| Automation | Schedule | Profile | Duration cap |
|---|---|---|---|
| `quobo-build-hourly` | hourly | quick | 30 min |
| `quobo-build-daily-full` | daily 03:30 local | full | 45 min |
| `quobo-build-weekly-docs` | Sun 04:00 | docs | 45 min |

Each run executes `scripts/continuous_build.ps1`, which self-reports to
[[Build Status]] / [[Build Log]] / `outputs/ci/`. On FAIL the agent diagnoses
from `outputs/ci/last_build.json` + pytest output and appends a triage note —
**it never edits source code unattended**.

## On a red build

1. Open `outputs/ci/last_build.json` → `summary` + failing test ids.
2. Reproduce: `.venv/Scripts/python -m pytest <test_id> -x --tb=long`.
3. Classify: real regression / flaky / env drift (check `constraints.txt`
   vs `pip list`).
4. Record in [[Issue Register]] (manual section) with fix or owner.
5. Never flip [[Build Status]] by hand — rerun the script.

## Long-running experiment runs

Full ablation (25 repeats × 7 arms) ≈ 11 min:
`.venv/Scripts/python -m src.quobo.run_experiment configs/experiment_6class.yaml`.
Manifest lifecycle protects partial runs ([[Provenance and Reproducibility]]).

## Data re-acquisition

- TACO images: `python -m src.quobo.data_prep <cfg>` or resumable
  `python -m scripts.fetch_missing`.
- Validate before use: `python -m scripts.validate_data` ([[data_validation]]).

Related: [[Continuous Build]] · [[Scripts]] · [[Test Suite]]
