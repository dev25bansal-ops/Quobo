---
tags: [glossary]
up: "[[00 Home]]"
---

# Glossary

| Term | Meaning in this project |
|---|---|
| **QUBO** | Quadratic Unconstrained Binary Optimization — the selection objective in [[selection]]; binary x_i = "keep PCA component i" |
| **Muecke QFS** | QUBO formulation used: `Q = -α·Σ I_i x_i + (1-α)·Σ R_ij x_i x_j` (relevance − redundancy), α binary-searched until exactly k features chosen |
| **SA** | Simulated annealing (dimod), the local solver; real annealer path = S3 ([[Issue Register]]) |
| **MI** | Mutual information between PCA features and class labels, quantile-discretized (mi_bins=16) |
| **mRMR** | minimum-Redundancy-Maximum-Relevance classic filter selector (arm F) |
| **QSVM / QSVC** | Quantum SVM with kernel methods; here `FidelityStatevectorKernel` + `ZZFeatureMap` ([[qsvm]]) |
| **ZZFeatureMap** | Entangling circuit mapping features to qubit states; `linear` entanglement chosen over `full` |
| **FSK** | Fidelity Statevector Kernel — classically simulable quantum kernel (statevector simulator, no real HW needed) |
| **GroupShuffleSplit** | Split strategy grouping crops by source photo — same photo never in train+test (LKG-002 fix) |
| **Leak-free protocol** | Per-split scaler+PCA fit; no transductive leakage; defines v2 numbers ([[Key Results]]) |
| **Arms A–F / Z** | random / PCA-order / MI / QUBO / LASSO / mRMR / Full-50 ceiling |
| **Holm correction** | Family-wise error control over the 78 paired comparisons |
| **PSI** | Pollution Severity Index — **project-defined relative index, unvalidated, not a health index** (PSI-002) |
| **TACO** | "Trash Annotations in Context" dataset: 1,500 photos, coarse+fine classes; here 6 coarse classes, 1,836 crops |
| **TrashNet** | Stanford cross-dataset replication target (OPEN-10) |
| **run manifest** | `experiments/<run_id>/run_manifest.json` — lifecycle + sha256 artifact inventory ([[Provenance and Reproducibility]]) |
| **test_ids** | Saved crop IDs per repeat — the authority for prediction→crop mapping, no split replay |
| **Wilson interval** | Confidence interval for small-n proportions, used in zone cleanliness ([[uncertainty]]) |
| **ONNX parity** | Exported edge pipeline matches sklearn head 100% on outputs |
| **QPU** | Quantum Processing Unit (real hardware; D-Wave / IBM paths, token-gated) |
