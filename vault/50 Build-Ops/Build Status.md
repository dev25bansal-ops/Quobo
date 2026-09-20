# Build Status

<!-- BEGIN AUTO-BUILD-STATUS -->
> [success] Last build: 2026-09-20 12:16 — PASS (profile: quick)

- **Status**: `PASS`
- **Profile**: quick
- **Tests**: 162 passed, 0 failed, 0 errored, 0 skipped
- **Ruff**: clean
- **Schema import**: ok
- **pip-audit**: not run (quick profile)
- **Duration**: 63.1s
- **Commit**: a7c8522
- **Summary**: 162 passed, 1 deselected, 10 warnings in 56.72s
- **History**: [[Build Log]] · machine-readable: `outputs/ci/last_build.json`
<!-- END AUTO-BUILD-STATUS -->

## Notes (manual section — not overwritten)

- Baseline established 2026-09-20 by a manual `pytest tests/ -q` run:
  163 passed, 11 benign warnings. See [[Test Suite]].
- Gate semantics: **tests are the gate**; ruff is advisory; a `FAIL` here means
  a real regression, triage via [[Issue Register]] and `outputs/ci/` logs.
