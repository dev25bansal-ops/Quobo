---
tags: [config, tooling]
up: "[[00 Home]]"
---

# Experiment Configs (`configs/`)

Validated by [[config]] (whitelist + pydantic, fail-fast). All values below are
the actual shipped settings (verified 2026-09-20).

## `experiment_6class.yaml` — the paper run

- `seed: 42`
- classes: bottle, can, cigarette, carton, cup, lid (drops `other` — 52% of
  crops, mixes 50 unrelated categories and drowns MI/QUBO objectives; kept
  classes 123–340 crops each)
- features: MobileNetV2 → pca_components **50**
- selection: k **8** (equal budget across arms)
- qubo: mi_bins 16, sa_sweeps 20000, sa_repeats 10
- qsvm: reps 1, entanglement **linear** (7 ZZ pairs vs 28 full — lower
  concentration risk), max_train_samples 400 (kernel is O(n²)),
  test_fraction 0.25, tune_rbf true
- experiment: n_repeats **25**, arms A–F + Z_full

## `experiment.yaml` — legacy 7-class

Includes `other`; n_repeats 5; arms A–D only; `baselines: [rbf_svm]`
(mandatory per Huang 2021 QSVM caveat). Superseded by the 6class run.

## `experiment_trashnet.yaml` — replication (OPEN-10)

TrashNet 6 classes (glass/paper/cardboard/plastic/metal/trash),
min_images_per_class 100, `crops_dir_override: data/processed/crops_trashnet`.
Harness is **complete** — the file's header comment ("requires a small
extension") predates the fix and is stale. Procedure in
`configs/README_cross_dataset.md`.

## `taco_taxonomy.csv`

leaf→coarse supercategory map consumed by `data_prep.load_supercat_map`.

## Rules

- Unknown keys raise — when adding a key, update **both** `config.KNOWN_SUBKEYS`
  and `config_schema` (manual sync, see [[config]]).
- Cross-field rule: `selection.k ≤ features.pca_components`.
