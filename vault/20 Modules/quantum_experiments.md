---
tags: [module, new, quantum]
path: src/quobo/quantum_experiments.py
up: "[[00 Home]]"
---

# quantum_experiments (new — uncommitted side track)

Independent A2/A3/A4 track exploring **quantum kernels beyond the main ablation**:
trainable feature-map alignment, kernel ablation, and real-sampler/IBM execution.
Does **not** feed the main pipeline ([[run_experiment]]).

## Components

- Split handling: `FeatureSplit` (dataclass, `fingerprint`, `center`),
  `load_split`, `smoke_split`, `prepare_split`, `_sample_indices`, `_canonical_ids`.
- Feature maps & kernels: `make_feature_map` (`_circuit_api`, `_validate_map`),
  `make_statevector_kernel`, `sampler_kernel` (shot-based), `ibm_kernel`
  (real backend via `scripts/ibm_kernel.py`), `overlap_circuits`,
  `aggregate_sampler_results`.
- Trainable alignment: `centered_target_alignment` (objective), `train_alignment`
  (parameterized feature map tuned toward class separation), `evaluate_kernel`,
  `train_kernel_experiment`.
- Ablation harness: `ablation_grid`, `run_ablation` (drives
  `scripts/kernel_ablation.py`), `package_versions` (environment stamping),
  `_matrix`, `_json_bytes`, `write_new_json` (provenance-style atomic writes).

## Related scripts

- `scripts/train_quantum_kernel.py` — train alignment A2/A3.
- `scripts/kernel_ablation.py` — grid ablation.
- `scripts/ibm_kernel.py` — IBM Quantum execution (needs `qiskit-ibm-provider` +
  IBM token; skip-green without, like `qpu_run.py` does for D-Wave).

Docs: `docs/quantum_experiments.md`. Tests: `tests/test_quantum_experiments.py`.

Related: [[qsvm]] · [[Issue Register]] (S3 QPU tokens)
