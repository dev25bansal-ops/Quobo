---
tags: [build, ops, automation]
up: "[[00 Home]]"
---

# Continuous Build (24/7)

The project runs a **continuous quality gate** via `scripts/continuous_build.ps1`,
scheduled 24/7 by a CodeBuddy automation. It never deploys anything — it verifies
the tree stays green and records the result here.

## What one build does

1. **Ruff** lint (`src/ scripts/ tests/`) — non-blocking advisory (logs count).
2. **Pytest** full suite (`tests/ -q`) — the gate. Baseline: 163 passed (~2:51).
3. **Config sanity**: import `config_schema` to catch pydantic/schema drift.
4. **pip-audit** (`full`/`docs` only, advisory, network-dependent) — mirrors CI;
   caught + fixed CVE-2026-86049 (jupyter-server) on 2026-09-20.
5. Writes machine-readable `outputs/ci/last_build.json` + timestamped log.
6. Updates [[Build Status]] (the callout on [[00 Home]]) and appends to [[Build Log]].

## Profiles

| Profile | Steps | Use |
|---|---|---|
| `quick` | ruff + pytest `-m "not integration"` (~fast loop) | high-frequency checks |
| `full` | ruff + **all** pytest + config import | the scheduled 24/7 build |
| `docs` | `full` + Sphinx HTML build | weekly / on docstring changes |

Manual run:
```powershell
.venv/Scripts/python -m pytest tests/ -q          # the gate
pwsh scripts/continuous_build.ps1 -Profile full   # full build + vault update
```

## Verification status (2026-09-20)

All three profiles executed green end to end:
`quick` 162/162 · `full` 163/163 · `docs` 163/163 + Sphinx HTML OK
(`docs/_build/html/index.html`). See [[Build Log]].

## Scheduling (24/7)

A recurring CodeBuddy automation (`quobo-continuous-build`) runs the `full`
profile on a fixed cadence and updates [[Build Status]] / [[Build Log]]. See
[[Runbook]] for the exact automation spec.

## Design choices

- **Advisory lint, blocking tests**: matches `ci.yml` (ruff runs but the gate is
  pytest; pip-audit non-blocking). Keeps noise low while catching real regressions.
- **Offline-safe**: the suite needs no network/weights ([[Test Suite]]), so a
  build can run unattended any time.
- **Idempotent artifacts**: results land in `outputs/ci/` (gitignored scratch),
  never mutate source.
- **Provenance-aware**: a build only flips [[Build Status]] to green on a real
  full-suite pass — it will not inherit the prior "not claimed" ambiguity.

Related: [[Build Status]] · [[Test Suite]] · [[Issue Register]]
