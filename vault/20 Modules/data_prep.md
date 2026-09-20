---
tags: [module]
path: src/quobo/data_prep.py
up: "[[00 Home]]"
---

# data_prep — Stage 1

Downloads TACO and builds the classification crop dataset. CLI:
`python -m src.quobo.data_prep configs/experiment_6class.yaml`.

## Key functions

- `download_file` — streaming GET with `MAX_DOWNLOAD_BYTES = 100 MiB` cap, self-deletes on overflow.
- `_validate_dataset_paths(coco, root, context)` — security gate: rejects absolute paths, drive letters, `..`, resolve-escapes (raises, never silently skips).
- `download_taco_images` — uses `flickr_640_url`, tolerates dead links (404/410 fail-fast per OPEN-3).
- `record_provenance` — writes `data/raw/annotations_provenance.json` (sha256/mtime/n_images/n_annotations; OPEN-6).
- `load_supercat_map` — leaf→coarse from `configs/taco_taxonomy.csv` (note: `cfg_classes` param unused).
- `build_crops` — groups by image, decodes once; bbox rescaled to actual downloaded size (`sx=W/rec["width"]`); 10% margin; <8px → `too_small`; output paths guarded by `is_relative_to`; classes below `min_images_per_class` removed via `shutil.rmtree`.
- `run_data_prep(cfg)` — entry; writes `data/processed/data_prep_stats.json`.

## Output

`data/processed/crops/<class>/<crop>.jpg` — 1,836 crops, 6 classes for the paper run.

## Dead code

`SUPERCAT_MAP_URL` constant is unused.

Related: [[features]] · [[data_validation]] · [[Provenance and Reproducibility]]
