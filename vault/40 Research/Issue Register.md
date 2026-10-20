---
tags: [research, issues]
up: "[[00 Home]]"
source: research/ISSUE_REGISTER.md
---

# Issue Register (summary)

Mirror of `research/ISSUE_REGISTER.md` (authoritative source). State 2026-09-20.
For the full audited catalog with severity/priority/effort/timeline see
**`research/ISSUES_CATALOG_2026-10-20.md`** — the current catalog (3 Low open
as backlog; 49 issues fixed across the 2026-10-18/19/20 hardening passes:
6 Critical, 22 High, 21 Medium). The older `research/ISSUES_CATALOG_2026-10-19.md`
and `research/ISSUE_CATALOG.md` are earlier snapshots.

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

### 2026-10-18/19 hardening pass (18 commits)

Security: HTML-injection in the dashboard `source_label` and path traversal via
`run_id` in `find_complete_run` (both fixed). Correctness: non-contiguous
integer labels produced wrong MI; ONNX parity compared bytes to str and
silently reported 0%; `assert`-based PCA parity check vanished under `-O`;
`confusion_analysis` consumed incomplete runs; `n_reps_done` overcounted a
crashed rep. Hardening: naive datetimes, `zip(strict=True)`, atomic writes
(`src/quobo/atomic.py`), TensorFlow pin alignment, `run_experiment --help` is
now inert (argparse).

## Open — externally blocked (nothing code-side)

| ID | Item | Blocker / action needed |
|---|---|---|
| OPEN-7 | CI never exercised | **remote now exists** (`origin` → github.com/dev25bansal-ops/Quobo); push to trigger `.github/workflows/ci.yml` |
| OPEN-9 | Real Gwalior map | need ward-boundary GeoJSON (GMC/Smart City SPV/GADM) → Folium swap |
| OPEN-10 | TrashNet replication | ~2GB download + extract; harness ready (`crops_dir_override` + `configs/experiment_trashnet.yaml`) |
| S3 | D-Wave QPU run | `DWAVE_API_TOKEN` from leap.dwavesys.com; `scripts/qpu_run.py` ready |
| PRV-004 | legacy runs not consumable | re-run or `--mode demo-ground-truth` |
| H-02 | no `v2.0-results` tag | only `v1.0-results` exists, and it predates the leak-free path — it cannot reproduce the v2 paper numbers. Tag HEAD after a clean end-to-end run. |

## Maintenance debt (minor)

- **SEC (2026-09-20)**: pip-audit found `jupyter-server 2.20.0` /
  CVE-2026-86049 → upgraded to 2.21.1, `constraints.txt` synced, audit now
  clean; advisory pip-audit step added to `full`/`docs` builds ([[Continuous Build]]).
- `pydantic` now declared in `pyproject.toml` (fixed 2026-09-20).
- `.gitignore` extended (2026-09-20): `outputs/`, `.workbuddy-ai/`, stale
  `experiment_final_e2e_log.txt`, and Obsidian per-user state
  (`vault/.obsidian/workspace*.json`, `graph.json`) — vault notes stay tracked.
- pre-commit hook installed (`pre-commit install`, ruff env cached). The hook's
  pytest gate now uses `./.venv/Scripts/python.exe` — the bare system Python
  lacked tensorflow and failed every commit at collection (fixed 2026-10-18).
- Dead code: `data_prep.SUPERCAT_MAP_URL` unused; `load_supercat_map(cfg_classes)`
  param unused; `qsvm.max_memory_mb` schema key has no consumer.
- `config.KNOWN_SUBKEYS` is now derived from `config_schema` at import time
  (no manual sync needed).
- **Open**: `qpu_run.py` still uses the legacy leaky `run_features` (H-01) and
  writes its result non-atomically (M-01) — both blocked on a test-mock refactor
  in `tests/test_qpu_run.py`.

Related: [[Build Status]] · [[Continuous Build]]
