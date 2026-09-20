---
tags: [module]
path: src/quobo/run_experiment.py
up: "[[00 Home]]"
---

# run_experiment — orchestrator

CLI: `python -m src.quobo.run_experiment configs/experiment_6class.yaml [--track]`.

## Flow (`run` → `_run_body`)

1. `load_config` + `validate_config` + `ensure_dirs` ([[config]]).
2. Create `experiments/<run_id>/`, write `run_manifest.json` with `status:"running"`.
3. Ensure Stage 1/2 artifacts ([[data_prep]], [[features]]).
4. For each repeat × arm: [[selection]] → [[qsvm]] → write `rep<N>_<arm>.json`
   (includes `test_ids`, `y_test`, predictions, α/trace/nudged_to_k).
5. `holm` + `paired_significance` → `results/tables/significance_*.csv`.
6. Manifest → `complete` with `{sha256, bytes}` `_artifact_inventory`; failures
   flip it to `failed` (never consumed downstream — [[Provenance and Reproducibility]]).

## Key helpers

- `_sha256_file`, `_artifact_inventory`, `_config_digest` — provenance.
- `_write_json_atomic` — no torn JSON.
- `crop_image_group` — group key for GroupShuffleSplit (same photo never in
  train+test; LKG-002 fix).
- `should_stop_early` — `experiment.early_stop` support.
- `log_to_mlflow` — optional `[track]` extra, off by default.
- `arm_label` — display names for A–F/Z.

## Statistics

n_repeats 5..100 (paper: 25), paired tests, Holm-corrected across 78 comparisons.

Related: [[Architecture]] · [[Test Suite]]
