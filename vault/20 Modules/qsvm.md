---
tags: [module]
path: src/quobo/qsvm.py
up: "[[00 Home]]"
---

# qsvm — Stage 4

Quantum + classical classifiers over the selected k features.

## Key functions

- `make_qsvc` — `ZZFeatureMap` + `FidelityStatevectorKernel` + `QSVC`
  (qiskit-machine-learning; statevector simulator, runs locally free).
- `fit_scale_for_feature_map` — scales features into the feature-map range
  (respects `qsvm.max_qsvm_features` ≤30 guard from [[config]]).
- `evaluate` — accuracy + per-repeat metrics for one classifier.
- `run_all_classifiers` — QSVC + RBF-SVM (+ linear, + tuned grid when
  `tune_rbf`); baselines configurable via `experiment.baselines`.

## Notes

- Entanglement: `linear|full` from `QSVMConfig`.
- `max_train_samples` (≥50) bounds statevector cost.
- Reported result: QSVC ≈37% (kernel parity — quantum kernel matches classical RBF
  on this data; see [[Key Results]]).
- Real-Hardware path: IBM sampler kernel + trainable alignment live in the side
  track [[quantum_experiments]], not here.

Related: [[selection]] · [[run_experiment]] · [[Scripts]]
