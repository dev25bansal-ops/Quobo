---
tags: [research, issues]
up: "[[00 Home]]"
source: research/ISSUE_REGISTER.md
---

# Issue Register (summary)

Mirror of `research/ISSUE_REGISTER.md` (authoritative source). State 2026-09-20.
For the full audited catalog with severity/priority/effort/timeline see
**`research/ISSUE_CATALOG.md`** (19 findings: 0 Critical, 3 High, 10 Medium, 6 Low;
security audit verified clean).

## Resolved highlights

- **QML-001** MinMaxScaler leaked test data → per-split fit (2 tests green).
- **LKG-002** same-photo crops in train+test → `GroupShuffleSplit` via `crop_image_group`.
- **OPEN-5** transductive PCA → leak-free protocol, 25-repeat rerun, paper v2 numbers ([[Key Results]]).
- **OPEN-6** TACO provenance → `annotations_provenance.json` (sha256 `ac1ec605…`, n=1500).
- **PRV-001/002/003** provenance hardening: run manifests + saved `test_ids` +
  dashboard refuses incomplete runs ([[Provenance and Reproducibility]]).
- **ONNX-001** export manifest now covers backbone with sha256 inventory.
- **MET-001** significance: 78 Holm-corrected paired comparisons @ n=25.
- **PKG-001** packaging: `pip install -e .` + `constraints.txt`.
- **OPEN-8** folder-mode zone bug found & fixed (zone = grandparent dir).

## Open — externally blocked (nothing code-side)

| ID | Item | Blocker / action needed |
|---|---|---|
| OPEN-7 | CI never exercised | no GitHub remote — create repo + push; `.github/workflows/ci.yml` ready (pydantic added 2026-09-20) |
| OPEN-9 | Real Gwalior map | need ward-boundary GeoJSON (GMC/Smart City SPV/GADM) → Folium swap |
| OPEN-10 | TrashNet replication | ~2GB download + extract; harness ready (`crops_dir_override` + `configs/experiment_trashnet.yaml`) |
| S3 | D-Wave QPU run | `DWAVE_API_TOKEN` from leap.dwavesys.com; `scripts/qpu_run.py` ready |
| PRV-004 | legacy runs not consumable | re-run or `--mode demo-ground-truth` |

## Maintenance debt (minor)

- **SEC (2026-09-20)**: pip-audit found `jupyter-server 2.20.0` /
  CVE-2026-86049 → upgraded to 2.21.1, `constraints.txt` synced, audit now
  clean; advisory pip-audit step added to `full`/`docs` builds ([[Continuous Build]]).
- `pydantic` now declared in `pyproject.toml` (fixed 2026-09-20).
- `.gitignore` extended (2026-09-20): `outputs/`, `.workbuddy-ai/`, stale
  `experiment_final_e2e_log.txt`, and Obsidian per-user state
  (`vault/.obsidian/workspace*.json`, `graph.json`) — vault notes stay tracked.
- pre-commit hook installed (`pre-commit install`, ruff env cached) — but the
  tree has **no git remote** (OPEN-7), so it only guards local commits.
- Dead code: `data_prep.SUPERCAT_MAP_URL` unused; `load_supercat_map(cfg_classes)`
  param unused; `qsvm.max_memory_mb` schema key has no consumer.
- `config.KNOWN_SUBKEYS` ↔ `config_schema` manual sync — watch on every config change.

Related: [[Build Status]] · [[Continuous Build]]
