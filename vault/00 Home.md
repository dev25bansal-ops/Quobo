---
tags: [moc, quobo]
created: 2026-09-20
---

# Quobo — Vault Home

> [!info] About this vault
> Knowledge base for **Quobo**: QUBO-based quantum feature selection + QSVM for environmental image classification (IDP3 project).
> Source repo: `d:/Quobo` · Paper line: [[Key Results]]

## Latest build

![[Build Status]]

## Map of Content

### Project
- [[Overview]] — what the project is, dataset, pipeline stages
- [[Architecture]] — module map + data flow
- [[Provenance and Reproducibility]] — manifests, fingerprints, leak-free protocol

### Modules (`src/quobo/`)
- [[config]] · [[config]] (schema) — load + fail-fast validation
- [[data_prep]] — Stage 1: TACO download + crops
- [[features]] — Stage 2: MobileNetV2 → per-split PCA
- [[selection]] — Stage 3: arms A–F + Full-50
- [[qsvm]] — Stage 4: QSVC + classical baselines
- [[run_experiment]] — orchestration, manifest, significance
- [[uncertainty]] — Wilson interval + vote agreement
- [[data_validation]] — offline data-contract checks
- [[quantum_experiments]] — A2/A3/A4 quantum-kernel side track

### Tooling
- [[Scripts]] — `scripts/` entry points
- [[Test Suite]] — 163 tests (green baseline 2026-09-20), markers, offline policy
- [[Configs]] — the three experiment YAMLs + taxonomy, and their rules

### Research
- [[Issue Register]] — resolved + externally-blocked issues
- [[Key Results]] — definitive leak-free v2 numbers
- [[Paper Status]] — paper draft state

### Build & Ops
- [[Continuous Build]] — how the 24/7 build works
- [[Build Log]] — append-only history (auto)
- [[Runbook]] — setup, common commands, troubleshooting
- [[ADR-001 Continuous Build Design]] · [[ADR-002 pydantic Declared as Core Dependency]]

### Reference
- [[Glossary]] — QUBO, QSVM, PSI, arms, provenance terms

## Conventions

Notes use YAML frontmatter + wikilinks. Module notes mirror `src/quobo/*.py`.
Auto-generated files ([[Build Status]], [[Build Log]]) are written by `scripts/continuous_build.ps1` — edit only the sections marked manual.
