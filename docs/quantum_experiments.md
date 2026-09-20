# Local quantum-kernel experiments (A2/A3/A4)

These are **new, opt-in experiments**, not re-runs of the frozen
`experiments/20260914_174024` baseline. Synthetic smoke fixtures are not TACO
measurements. No novelty, causal effect, or quantum advantage is established by
these tools. No IBM hardware experiment has been run as part of implementation.

## Environment and safe entry points

Use the project's installed virtual environment from the repository root:

```bash
./.venv/Scripts/python.exe scripts/train_quantum_kernel.py --help
./.venv/Scripts/python.exe scripts/kernel_ablation.py --help
./.venv/Scripts/python.exe scripts/ibm_kernel.py --help
```

Help performs no feature extraction or network access. Training and ablations
require Qiskit >=2.1 and Qiskit Machine Learning >=0.9 for execution; missing
optional imports produce actionable errors. Installed APIs were inspected before
implementation: Qiskit 2.5.2, QML 0.9.1 include
`TrainableFidelityStatevectorKernel` and the function-form `zz_feature_map`.
**There is no assumed `QuantumKernelAlignmentLoss` class.** The centered alignment
objective and first-order SPSA loop are explicit implementations in
`src/quobo/quantum_experiments.py`.

IBM execution additionally requires the optional `qiskit-ibm-runtime` package.
It was **not installed** in the development environment; its service adapter is
mock-tested, while circuits, transpilation, and local `StatevectorSampler` are
exercised against installed Qiskit. This is not a claim of hardware validation.

## Input contract and leakage protection

Training/ablation inputs are pickle-free NPZ files with these exact keys:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `X_train`, `X_test` | finite float matrices, equal width | candidate features |
| `y_train`, `y_test` | 1D numeric or string arrays | categorical class labels |
| `groups_train`, `groups_test` | 1D numeric or string arrays | original source image IDs |
| `ids_train`, `ids_test` | 1D globally unique arrays | individual crop/sample IDs |
| `transform_fit` | scalar string | `raw` or `train_only` |

Arrays must have matching row counts. Groups must be disjoint across train/test;
all sibling crops from one source image must use the **same group**, not a crop ID.
Sample IDs must be unique across both splits. At least two train classes are
required; unseen test classes are rejected. Numeric labels are treated as
categories, not ordinal regression targets.

**The caller must create the group-disjoint split before any data-dependent
preprocessing.** `transform_fit` is an auditable declaration, not an automatic
upstream leakage detector. Use raw MobileNet embeddings, or PCA/scaling fitted on
train only. Do not hand in the globally fitted legacy feature CSV. The raw
baseline NPZ with `embeddings`, `labels`, `crop_paths` is deliberately not accepted
as if it were already a safe split. A downstream exporter should derive original
image groups from verified crop provenance, split groups, optionally fit PCA on
train, then write this contract to a **new** path.

Each run applies a seeded train cap retaining every class, then ranks candidate
features using `mutual_info_classif` on that capped training population only.
This implementation offers the MI selector, not all seven historical selectors.
Selected features are scaled to `[0.05, 1.52]` with a train-fitted MinMaxScaler.
Test values are transformed with the same scaler, **not clipped or refitted**;
out-of-range held-out values may produce angle aliasing. The test cap is seeded
and label-independent. Exact sample indices, IDs, groups, selected feature indices,
MI scores, scaler extrema, split content hash, seed, and package versions are
recorded. Both fixed and trained comparisons use identical capped populations.

## A2: trainable target-alignment kernel

```bash
# Cheap synthetic real-statevector execution: <=2 SPSA iterations, 12 train/6 test
./.venv/Scripts/python.exe scripts/train_quantum_kernel.py --smoke --output scratch/new-training-smoke.json

# No execution; inspect the plan
./.venv/Scripts/python.exe scripts/train_quantum_kernel.py --input scratch/split.npz --dry-run

# Explicit bounded local experiment on an existing safe split
./.venv/Scripts/python.exe scripts/train_quantum_kernel.py --input scratch/split.npz --execute --budget 4 --iterations 10 --max-train 48 --max-test 64 --output scratch/new-training.json
```

Without `--execute` or `--smoke`, training only prints a plan. An explicit
`--dry-run` overrides smoke execution. Output JSON is created exclusively; an
existing path is rejected rather than overwritten.

The feature map replaces each ZZ input `x_j` by `theta_j * x_j`. Trainable
weights are **inside the data encoding** and therefore change pairwise fidelity.
A common final unitary would cancel from fidelity and is intentionally not used.
At `theta = ones`, the map equals the fixed ZZ kernel. This equality and genuine
parameter dependence have real statevector tests.

The objective is centered Frobenius target alignment:

- `T[i,j] = 1` when training class labels match, otherwise `0`.
- Center both `K` and `T` by subtracting row/column means and adding the global mean.
- Maximize `<K_centered,T_centered> / (||K_centered|| ||T_centered||)`.

This is the centered one-hot class Gram target, valid for multiclass labels. It
is invariant to renaming or recoding class labels. A constant kernel scores zero.

The optimizer is **explicit first-order SPSA**, with seeded Rademacher directions,
`a_t = learning_rate / t^0.602`, `c_t = perturbation / t^0.101`, initial weights
all one, and weights clipped to `[0.1,3]`. Defaults are learning rate `0.2` and
perturbation `0.1`. The loop records initial, plus/minus, and iterate evaluations;
the best training-objective checkpoint among **all** evaluated points is restored,
including the initial fixed kernel. This may restore a perturbation probe rather
than the final iterate. It cannot promise improved held-out performance.

Training is bounded to 0–100 steps and at most 128 train samples (default 48),
with exactly `1 + 3*iterations` objective evaluations. Test cap is 1–256
(default 64); qubits/features are bounded to 2–12, reps 1–3. A precomputed-kernel
SVC with balanced classes and fixed `C=1` evaluates both fixed and restored
kernels on held-out data. **C is not tuned**, and held-out labels never enter
selection, scaling, or kernel optimization. Reported scores are descriptive,
not confidence intervals or evidence of quantum speedup.

## A3: reproducible fixed-kernel ablations

```bash
# Write a 24-cell manifest only: 3 reps x 2 entanglements x 4 budgets
./.venv/Scripts/python.exe scripts/kernel_ablation.py --dry-run --output scratch/ablation-plan

# Execute one cheap synthetic cell, retain manifest for the requested grid
./.venv/Scripts/python.exe scripts/kernel_ablation.py --smoke --output scratch/ablation-smoke

# One missing local cell per invocation by default; repeat to resume
./.venv/Scripts/python.exe scripts/kernel_ablation.py --input scratch/split.npz --execute --reps 1 2 3 --entanglement linear full --budget 4 6 8 12 --output scratch/ablation
```

`--max-runs` explicitly raises the missing-cell execution limit from 1 to at most
24. No full multi-hour sweep starts by default. Smoke always executes at most
one missing cell and caps samples to 12/6. Dry runs without `--input` use a marked
synthetic fixture solely to make a manifest.

The configurable grid is `reps={1,2,3}`, `entanglement={linear,full}`,
`budget={4,6,8,12}`. These are fixed ZZ kernels, **not trained kernels**.
Here `reps` means **circuit repetitions**, not statistical replicates. To inspect
seed sensitivity, run explicit separate seeds; no significance test is provided.
No winning cell is chosen from held-out scores, and a later user selection of a
winner would require another untouched evaluation set.

Each result key hashes the full input fingerprint, map configuration, seed,
sample caps, implementation version, preprocessing protocol, and package versions.
Manifests are separately content-addressed. Resume verifies key/specification and
completion fields before reusing results; inconsistent/partial JSON errors rather
than being silently reused. Move a corrupt new result aside before retrying.
All writes are exclusive creates, so existing frozen outputs are not overwritten.

## A4: IBM SamplerV2 fidelity harness

```bash
# Safe, credential-free plan: builds three 2-qubit overlap circuits locally
./.venv/Scripts/python.exe scripts/ibm_kernel.py --dry-run

# This fails explicitly: ZNE is unsupported, including in dry-run mode
./.venv/Scripts/python.exe scripts/ibm_kernel.py --dry-run --mitigation zne
```

Input NPZ for this lower-level harness uses **`X` and optional `Y`**, already
angle-scaled using an appropriate train-only preprocessing protocol. It does
not select features, load labels, train an SVC, or infer source groups. Supply
`X=train` for a symmetric train Gram matrix, then `X=test,Y=train` for a
rectangular cross-kernel, retaining external row IDs/provenance. Limit is 32 rows
per side and 2–12 qubits.

For each pair it measures `U(y)^dagger U(x)` and estimates squared fidelity by
`P(all measured bits = 0)`. Symmetric mode runs the upper triangle **including
self-pairs** and mirrors it; noisy diagonal values are **not forced to one**.
Rectangular mode runs all pairs. The JSON contains empirical frequencies, pair
indices, requested and observed shots, backend/instance, job ID, transpilation
seed, map configuration, input hash, circuit count, package versions, and flags
indicating no PSD repair and no mitigation. Raw noisy matrices can be indefinite;
this harness does not pretend that PSD projection is hardware error mitigation.

There is **no implicit network, saved-account search, environment-token lookup,
or simulator fallback**. To execute in a future separately authorized session,
all of these are mandatory together:

- `--execute`
- `--input <overlap-input.npz>`
- `--backend <explicit-backend>`
- `--instance <explicit-instance-CRN>`
- `--prompt-token` (interactive hidden prompt, token not saved or serialized)

The library equivalent also requires an explicit token/backend/instance and
`execute=True`. Hardware execution may incur provider usage/cost; it was not
performed here. `--seed` controls **transpilation only**, not device randomness.
Shots are bounded to 1–100,000 per circuit (default 1024).

**Supported mitigation is `none` only. Sampler ZNE is not implemented.** Requests
for `zne` or any other mitigation error before credential discovery or submission.
No Estimator-only resilience option is advertised as Sampler support.

## Verification

Run only this track's standalone tests:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_quantum_experiments.py -q
./.venv/Scripts/python.exe -m ruff check src/quobo/quantum_experiments.py scripts/train_quantum_kernel.py scripts/kernel_ablation.py scripts/ibm_kernel.py tests/test_quantum_experiments.py
```

Coverage includes multiclass label invariance, train-only preprocessing, group
leakage rejection, sample-ID preservation, real parameter-dependent statevectors,
SPSA learning/determinism/best restore, fixed-vs-trained held-out evaluation,
keyed resume, exclusive output writes, local overlap probabilities, real local
Sampler bit counts, rectangular assembly, noisy diagonals, mocked IBM adapter
plus installed transpiler, and credential-free CLI help/dry runs. Quantum tests
are dependency-gated if Qiskit/QML is absent; a skip is not execution success.
