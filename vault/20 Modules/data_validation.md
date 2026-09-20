---
tags: [module, new]
path: src/quobo/data_validation.py
up: "[[00 Home]]"
---

# data_validation (new — uncommitted)

Offline **data-contract validator**: audits `data/` before a run so a corrupt
dataset fails loudly instead of poisoning results. Entry:
`scripts/validate_data.py` → `validate_data(root, cfg)`.

## Design

- `ValidationReport` — accumulates errors (`error`, `unique`, `bad_constant`,
  `scan_error`), serializable via `as_dict`.
- Structural guards: `_portable_parts`, `_contained` (path-traversal), `_integer`,
  `_finite` (JSON sanity).
- Checks:
  - `_validate_coco` — TACO annotation schema, id uniqueness, bbox sanity.
  - `_validate_provenance` — `annotations_provenance.json` matches actual files (sha256).
  - `_verify_image` / `_scan_images` — decodable images, size bounds.
  - `_crop_inventory` / `_validate_crop_meta` — crops consistent with meta;
    ⚠️ **mirrors** `features.crops_fingerprint` formula (implicit coupling).
  - `_read_npz` / `_validate_cache` — feature-cache shape/fingerprint integrity.

## Tests

Covered by `tests/test_data_validation.py` + `tests/test_data_security.py`
(path-traversal / hostile-input cases) — see [[Test Suite]].

Related: [[features]] · [[Provenance and Reproducibility]] · [[Scripts]]
