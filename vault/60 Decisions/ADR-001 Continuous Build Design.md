---
tags: [adr, build]
status: accepted
date: 2026-09-20
up: "[[Continuous Build]]"
---

# ADR-001: Continuous build = scheduled gate, not deployer

## Context
Quobo is a research repo with a green 163-test suite, no GitHub remote (OPEN-7),
and a paper whose credibility rests on provenance. "24/7 build" must not risk
mutating research artifacts.

## Decision
- `scripts/continuous_build.ps1` is **verify-only**: ruff (advisory) + schema
  import + pytest (gate) + optional Sphinx. Never deploys, never commits,
  never edits source.
- Three CodeBuddy automations: hourly `quick`, daily 03:30 `full`,
  weekly Sun `docs`. Lock file prevents overlap; stale lock > 30 min ignored.
- Results land in `outputs/ci/` (json+csv) and the vault ([[Build Status]],
  [[Build Log]]). Failure triage is appended to the status note's manual
  section by the agent; classification = regression / flaky / env drift.
- Tests are the only blocking gate — mirrors `.github/workflows/ci.yml`
  philosophy (ruff advisory, pip-audit non-blocking).

## Consequences
- Unattended runs are safe on research data; the vault is the single pane of glass.
- CI parity: local builds cover everything `ci.yml` does except pip-audit
  (run it manually before releases) and ubuntu coverage (Windows-only local loop).
- No remote → no push CI yet; local automation is the interim 24/7 layer
  (OPEN-7 ready to retire once a GitHub repo exists).

Rejected alternatives: git-push-triggered Actions only (no remote); always-on
loop script (fragile across reboots); auto-fix agent (unsafe for research code).
