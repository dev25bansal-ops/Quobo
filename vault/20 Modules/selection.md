---
tags: [module]
path: src/quobo/selection.py
up: "[[00 Home]]"
---

# selection — Stage 3 (arms A–F + Z)

Selects k=8 features from the 50-d PCA space, one function per arm, dispatched by
`select_features(arm, ...)`.

## Arms

| Arm | Function | Method |
|---|---|---|
| A | `select_random` | random k (seeded) |
| B | `select_pca_order` | first-k PCA components |
| C | `select_mi` | mutual-information top-k (`_discretize`, `compute_mi_table`, mi_bins) |
| D | `select_qubo` | **QUBO: MI-relevance minus redundancy → simulated annealing** |
| E | `select_lasso` | L1 logistic coefficients top-k |
| F | `select_mrmr` | minimum redundancy maximum relevance |
| Z | (Full-50) | no selection ceiling (`no_selection: true` in rep JSONs) |

## QUBO formulation

- `qubo_matrix` — builds Q from the MI table: diagonal = relevance, off-diagonal =
  −redundancy penalty; cardinality handled by λ nudging (`nudged_to_k` recorded).
- `solve` — dimod simulated annealing (`sa_sweeps`, `sa_repeats` from
  [[config|QUBOConfig]]); `num_reads = repeats` (PERF-001 fix: removed 5× wasted reads).
- Deterministic given seed; α (redundancy weight) + trace persisted in rep JSONs (OPEN-2 fix).

## Caveats

- MI discretization bias documented in paper §5 (MI-BIN).
- Under the strict leak-free protocol all principled arms **tie statistically** —
  QUBO's value is the globally-optimal formulation portable to real annealers
  ([[quantum_experiments]], [[Key Results]]).

Related: [[features]] · [[qsvm]] · [[run_experiment]]
