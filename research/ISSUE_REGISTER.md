# Quobo — Issue Register (Post-Remediation)

**Date:** 2026-09-02 · **State:** after executing AUDIT_REPORT.md through tag `v1.0-results` (11 commits)
**Method:** every item re-verified against the current code/artifacts (not carried over from the pre-fix audit); resolved items listed with their verification evidence, open items with repro, severity, effort, dependencies, timeline.

---

## Part 1 — Resolved (verified this session)

| ID | Issue (was) | Sev | Fix commit | Verification |
|---|---|---|---|---|
| VCS-001 | No git repo | Critical | 5d889c9 | 11 commits, `v1.0-results` tag, clean tree |
| QML-001 | MinMaxScaler fit on test (QSVM leak) | High | 5d889c9 | `test_scaler_params_from_train_only` + `test_scaler_identical_transform_of_same_point` green; QSVM accuracy rose 34.1→37.4% after fix |
| PSI-001 | PSI ignored documented hazard weights | High | 5d889c9 | `test_psi_hazard_weight_multiplies` green; dashboard bands now spread 114→500 |
| MET-001 | No significance tests, n=5 | High | 5d889c9 | `significance_20260902_034140.csv`: 63 Holm-corrected comparisons, 25 repeats |
| LKG-002 | Same-photo crops straddled train/test | High | 011b099 | GroupShuffleSplit; `test_selection_deterministic` + integration test use groups |
| DOC-001 | README crash + phantom modules | Medium | 5d889c9 | All documented commands re-executed successfully this session |
| CCH-001 | Unconditional stale feature cache | Medium | 5d889c9 | Fingerprint (hash + per-class counts) validated on every load — confirmed matching live |
| DRY-001 / SEED-001 | Duplicated retraining + fixed subsample seed | Medium | 011b099, d18486e | Confusion analysis consumes persisted predictions (log: "using run 20260902_034140, no retraining") |
| PERF-001 | SA 5× wasted reads (0.54s vs 0.11s/call) | Medium | 5d889c9 | `num_reads=repeats`; QUBO arm ~3s/rep (was ~10s) |
| BBOX-001 | Clamp in original-image space | Low | 5d889c9 | `test_bbox_clamp_actual_image_dims` — analytic 48×96 case passes |
| PKG-001 | No packaging, sys.path hacks, unpinned deps | Medium | 011b099, cf8d444 | `pip install -e .` works; constraints.txt (150 pins) |
| SEC-001/2/3 | Download caps / path escape / HTML injection | Low | 5d889c9 | Size cap + `is_relative_to` guard + `html.escape` at all 3 interpolation sites |
| TST-001 | Tautological smoke test only | High (debt) | c656ab6 | 14 tests (unit + integration), all green; ruff clean |
| MI-BIN | MI bias caveat | Low | — | Documented in PAPER_DRAFT.md §5 |

**Residual note (honest accounting):** the on-disk crops (built 2026-08-26) predate the BBOX-001 fix. At 640px Flickr sizes the clamp bug had near-zero effect (verifier: real-data impact ≈0, code defect real), but a clean-room rebuild of crops + features would make the provenance airtight. Cost: ~4 min recompute. Decision deferred to you — numbers would shift within noise.

---

## Part 2 — Open issues (verified current, with repro)

### OPEN-1 · numpy spec contradicts the pinned environment — **High** · P1 · 0.5h · no deps
**Repro:** `grep numpy requirements.txt pyproject.toml` → `numpy<2.3`; `python -c "import numpy"` → **2.5.2**.
**Expected vs actual:** any fresh `pip install -r requirements.txt` resolves numpy <2.3, but v1.0 results were produced on 2.5.2 (constraints.txt). A new machine may fail to even import the env (TF 2.21 requires numpy≥2.3-era ABI in practice) or silently produce a different-ABI run.
**Why it matters:** reproducibility claim in the paper depends on constraints.txt being *the* spec; requirements/pyproject advertise a different env.
**Fix:** raise both to `numpy>=2.3,<2.6` (or exactly `==2.5.2`) to match constraints.txt. **Timeline: today.**

### OPEN-2 · `selection_seconds` for Z_full is fake; alpha never persisted — **Medium** · P2 · 1h
**Repro:** run any experiment; open `experiments/<ts>/rep0_Z_full.json` → `selection_seconds` is the time to build `range(50)` (~0µs, reported as a real stage), `mi_table` is null. Arm-D rep JSONs record `I`/`R` tables but **not the α the bisection converged to** (audit MET-006 recommendation, unimplemented).
**Expected vs actual:** selection bookkeeping should either be omitted for no-selection arms or record `{"no_selection": true}`; QUBO arm should persist α + per-α subset sizes for the stability figure.
**Impact:** a reviewer recomputing "selection cost" from the CSV gets a meaningless 0.0 for the ceiling arm; α reproducibility claims have no artifact. **Depends:** nothing. **Timeline: this week** (bundle with next rerun — 0 cost if the rerun happens anyway).

### OPEN-3 · `fetch_missing.py` reads unbounded response into memory — **Medium** · P2 · 0.5h
**Repro:** `scripts/fetch_missing.py:37` — `Image.open(io.BytesIO(r.content))` with no `len(r.content)` check. data_prep's `download_file` got the SEC-001 cap (verified) but this second fetcher (added later for the 464 fallback images) did not.
**Expected:** cap ~20 MB per image, matching the documented fix. **Impact:** a hostile/compromised Flickr-mirror response could OOM the process; also trusts Pillow's decompression bomb guard implicitly.
**Depends:** none. **Timeline: this week.**

### OPEN-4 · Dead config keys still advertised — **Medium** · P2 · 0.5h
**Repro:** `configs/experiment_6class.yaml` defines `img_size`, `standardize`, `qsvm.feature_map` — **0 usages in src/ or scripts/** (verified by grep). `data.min_images_per_class` is used only by data_prep, not by the experiment path. Also `experiment.baselines` was removed from 6class.yaml but the legacy `configs/experiment.yaml` still carries divergent keys.
**Expected vs actual:** changing `feature_map: ZZFeatureMap` to anything has zero effect — a silent no-op trap for your teammates. **Fix:** delete the dead keys or wire them; add an unknown-key warning in `load_config` (fail-loud beats fail-silent).
**Depends:** none. **Timeline: this week** (with OPEN-4's natural partner — collapsing the two YAMLs into one canonical config + `configs/paper.yaml` snapshot as planned).

### OPEN-5 · Transductive PCA (LKG-001) — **Medium (accepted risk)** · P3 · 4–8h refactor
**Repro:** `features.py:104-109` — StandardScaler + PCA fit on all 1,836 crops pre-split; docstring documents it as a known limitation. Uniform across arms, so rankings/significance survive, but absolute accuracies are optimistic by an unknown margin.
**Fix path (planned but not done):** cache raw 1280-d embeddings alongside the PCA CSV; refit scaler+PCA inside each split in `run_experiment`. **Business impact:** converts the paper's §5 limitation paragraph into a strength (leak-free protocol). **Depends:** one full rerun (~25 min at 7 arms). **Timeline: before submission.**

### OPEN-6 · TACO snapshot provenance unrecorded (PROV) — **Medium** · P3 · 1h
**Repro:** `data_prep.py` downloads `annotations.json` from GitHub master; no URL+sha256+download-date stored anywhere (data_prep_stats.json has counts only). The survey documented v1.0 (1,500) vs living dump (3,831) divergence — which snapshot you used is currently unrecoverable without the file's mtime.
**Fix:** record `{"source": TACO_URL, "sha256": ..., "downloaded": date, "n_images": 1500}` at download time. **Depends:** none. **Timeline: before submission** (the paper must state its snapshot).

### OPEN-7 · CI is unexercised — **Low** · P3 · 0.5h + a push
`ci.yml` exists (ruff + 14 tests, ubuntu py3.12) and the local equivalents pass, but the repo has no remote — it has never run on GitHub. First `git push` will reveal any Linux-only breakage (path separators, `Scripts/` vs `bin/` assumptions in docs).
**Depends:** you creating the GitHub repo. **Timeline: with publication packaging.**

### OPEN-8 · `zone_of` folder-mode + predictions-CSV mode are implemented but untested on real data — **Low** · P3 · 2h
**Repro:** `load_crops(mode="folder")` and the `predictions.csv` ingestion path exist; tests cover `compute_zone_table` on prediction-style frames but no end-to-end test drives folder mode (no real Gwalior ward data exists yet).
**Impact:** the "real deployment" claims in README/paper rest on unit-level evidence only. **Depends:** ward data. **Timeline: when data arrives.**

### OPEN-9 · Dashboard map is a CSS grid, not real geography — **Low (design)** · P3 · 0.5–2h
S6 (Folium choropleth of Gwalior wards) was audited as high-impact for civic stakeholders; current dashboard still renders the 2×3 demo grid. The analytics are correct; the visual is demo-grade. **Depends:** ward boundary GeoJSON (you). **Timeline: when GeoJSON arrives.**

### OPEN-10 · Cross-dataset replication (TrashBox/TrashNet) not run — **Low (scope)** · P4 · 1–2 days
Pre-empts the "single-dataset result" rejection. The pipeline is dataset-agnostic (crops → folders); only TrashBox licensing (none stated) blocks redistribution, not local analysis. **Depends:** download + one rerun per dataset. **Timeline: pre-submission stretch goal.**

---

## Part 3 — Verification matrix

| Check | Status | Evidence |
|---|---|---|
| 14/14 pytest | ✅ | run this session, 6.1s |
| ruff clean | ✅ | "All checks passed" |
| Reproducibility | ✅ | same seed → same selection `[0,1,5,7,16,21,33,40]` |
| Downstream consumers (no retrain) | ✅ | confusion + dashboard both re-ran from persisted JSONs |
| Fresh-env install matches paper env | ❌ | **OPEN-1** — requirements/pyproject vs constraints.txt disagree |
| Selection bookkeeping complete | ❌ | **OPEN-2** |
| Size caps on all fetchers | ❌ | **OPEN-3** |
| Config keys all live | ❌ | **OPEN-4** |
| Leak-free PCA | ❌ accepted | **OPEN-5** documented |
| Dataset provenance | ❌ | **OPEN-6** |

**Priority-ordered remediation queue:** OPEN-1 (today) → OPEN-3, OPEN-4, OPEN-2 (this week, ~2.5h total) → OPEN-5, OPEN-6 (pre-submission, one rerun) → OPEN-7 (first push) → OPEN-8, OPEN-9, OPEN-10 (data-dependent).
