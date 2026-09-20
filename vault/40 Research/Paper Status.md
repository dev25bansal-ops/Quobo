---
tags: [research, paper]
up: "[[00 Home]]"
---

# Paper & Research Docs

Authoritative sources live in `research/`:

| File | Role |
|---|---|
| `PAPER_DRAFT.md` | the paper — on **leak-free v2 numbers** ([[Key Results]]), with v1-vs-v2 protocol materiality reported transparently |
| `RESEARCH_REPORT.md` | background research (quantum stack rationale, pinned per findings) |
| `AUDIT_REPORT.md` | original audit (tag `v1.0-results`); most items resolved — see [[Issue Register]] |
| `ISSUE_REGISTER.md` | post-remediation issue queue (v3, 2026-09-17) |
| `_raw/` | 25 raw JSON research captures (sources/evidence) |
| `audit/` | audit working files |

## Paper spine

- **Method**: TACO → MobileNetV2 → leak-free per-split PCA(50) → QUBO(8) → QSVM;
  7-arm ablation + Full-50 ceiling; 25 group-aware repeats; Holm-corrected paired tests.
- **Headline finding**: principled arms tie; QUBO ships a globally-optimal
  formulation portable to annealers ([[Key Results]]).
- **PSI positioning**: project-defined relative index, **not a health index**
  (PSI-002) — must stay labeled this way in any figure/text.

## Open research threads

- Real-QPU EXP-D (S3, needs D-Wave token).
- IBM / trainable-kernel track ([[quantum_experiments]]).
- Cross-dataset replication on TrashNet (OPEN-10, harness ready).
- Real Gwalior ward map (OPEN-9, needs GeoJSON).

Related: [[Issue Register]] · [[Overview]]
