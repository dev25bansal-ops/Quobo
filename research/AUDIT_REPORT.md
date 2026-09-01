# Quobo — Comprehensive Project Audit & Strategic Roadmap

**Project:** IDP3 Quobo — QUBO feature selection + QSVM for waste-image classification
**Audit method:** 20-agent workflow (6 audit lenses + strategy analyst + 13 adversarial verifiers); every Critical/High finding below was independently re-verified against the code and runtime artifacts. 57 findings + 16 strategic opportunities, all traceable to `research/audit/*.json`.
**Codebase audited:** `src/quobo/` (7 modules, ~700 LOC) + `scripts/` (3 entry points) + `tests/` (1 smoke test) + configs, README, results from the 20260826_133947 full-data run (1,836 crops, 6 classes, 5 repeats).

**Headline experimental numbers (verified against `results/tables/summary_20260826_133947.csv`):**
RBF-SVM accuracy: QUBO-8 41.35±2.14 | PCA-8 40.39±1.51 | MI-8 38.87±3.99 | Random-8 31.98±3.85
QSVM accuracy: PCA-8 36.08 | MI-8 34.60 | QUBO-8 34.12 | Random-8 24.10

---

# 1. Project Analysis & Strategic Opportunities

## 1.1 Where Quobo stands

The research report (research/RESEARCH_REPORT.md) verified three firsts as of Aug 2026: (1) no published work combines QUBO feature selection with a quantum SVM on waste imagery; (2) no "Pollution Severity Index" for litter exists in indexed literature; (3) no quantum+GIS waste analytics chain exists anywhere. The pipeline works end-to-end and has citable numbers. The competitive position is real but **currently unbanked** — nothing is version-controlled, published, or timestamped, so the novelty window can close with no proof Quobo was first.

## 1.2 The six highest-value strategic moves (from 16 audited opportunities)

**S1. Close the CV→PSI loop — dashboard must consume model predictions (effort: S, impact: L)**
The dashboard currently reads ground-truth class folders, not classifier outputs. The "first quantum+GIS chain" claim is disconnected at exactly the joint a reviewer or municipal stakeholder would probe. Fix: a `--from-predictions` mode that reads a predictions CSV (crop, predicted_class, zone) and feeds the same zone analytics. This converts two deliverables into one integrated system.

**S2. Bank the novelty: git + pinned repo + arXiv preprint (effort: S, impact: M)**
Timestamps all three firsts. One-command reproducibility with a passing test suite is itself a differentiator — the research survey found most quantum-image repos don't run.

**S3. One real D-Wave run for EXP D (effort: S, impact: M)**
Upgrades the claim from "quantum-inspired annealing" to "formulated as a QUBO and solved on a quantum annealer" — factually true after a single Leap Hybrid run at n=50, and materially differentiating: no Indian environmental-QML work has done this.

**S4. Reviewer-proof baseline suite (effort: M, impact: L)**
Add LASSO and mRMR selection arms, a tuned RBF-SVM (grid over C/gamma — seconds on 400×8 data), and the full-50-feature ceiling. Directly answers "is QUBO better than cheap convex selection?" and neutralizes the Huang-2021 classical-parity objection before it is raised.

**S5. Statistical rigor: 25+ repeats with paired tests (effort: S, impact: L)**
QUBO-8 vs MI-8 (41.35 vs 38.87) and vs PCA-8 (41.35 vs 40.39) are within seed noise at n=5. Wilcoxon signed-rank on 25 paired repeats converts "we observed an edge" into "p<0.05" — or an honest parity statement, which the literature (SIGIR22 et al.) says is the likely truth and rewards.

**S6. Real Gwalior geospatial layer (effort: M, impact: M)**
Replace the 2×3 CSS grid with a Folium choropleth of actual Gwalior ward boundaries. For civic stakeholders a recognizable map is the demo; for reviewers it makes the first-quantum-GIS claim visually undeniable, pinned to the city whose WHO-#2-polluted ranking motivates the project.

**Further audited opportunities (detail in research/audit/strategy.json):** uncertainty-aware PSI with classifier-error confidence bands (a genuine methods first — no air-quality index propagates upstream classifier uncertainty); kernel-design ablation mapping the 8-qubit sweet spot empirically (operationalizes Schuld's "the encoding is the model"); cross-dataset replication on TrashNet/TrashBox (pre-empts single-dataset rejection, and TrashBox reuse sharpens the contrast with Kumsetty 2022, the closest Indian prior art); edge-latency benchmark of the 8-feature head (ONNX/TFLite — a deployment story orthogonal to the parity debate); OpenLitterMap/Debris-Tracker ingestion to replace the demo batch→zone mapping; NCAP/NAQI correlation plumbing to validate PSI against MPPCB air stations; a Swachh Survekshan positioning brief for Gwalior Municipal Corporation / Smart City SPV; venue-sequenced publication plan (FRUCT → IJCNN → IEEE QCE, exploiting the FRUCT lineage of the nearest prior paper).

---

# 2. Issues & Required Fixes (Verified Catalog)

All 13 Critical/High findings were independently re-verified by a dedicated verifier agent (runtime reproduction where feasible); verdicts recorded in `research/audit/verified_critical.json`. 13/13 confirmed, with several corrections to the auditors' numbers noted inline.

## Priority P0 — before any further results are generated

### VCS-001 · No git repository, no .gitignore — [CRITICAL] · 0.5–1h · no deps
**Problem:** `D:\Quobo` is not version-controlled. The code that produced the headline numbers has zero history/rollback; a naive first `git add .` would sweep in .venv (multi-GB TensorFlow), 252 MB of images, crops, and scratch dirs (sizes verified: __pycache__ 734K/928K/552K).
**Repro:** `ls -a` shows no `.git`; environment metadata confirms "Is directory a git repo: No".
**Fix:** `git init` + `.gitignore` (`.venv/`, `data/`, `experiments/`, `results/`, `*.log`, `__pycache__/`, scratch dirs, WhatsApp images) + initial commit of src/configs/scripts/tests/README. **Timeline: today.**

### QML-001 · MinMaxScaler fit separately on train and test — [HIGH, runtime-verified] · 1–2h fix + ~15min rerun · gates all QSVM numbers
**Problem:** `qsvm.py:52-59,68-69` — `scale_for_feature_map()` calls `fit_transform` on Xtr and again on Xte. Test-set statistics leak into the angle encoding, and train/test are mapped by *different* affine transforms, so the QSVC kernel compares geometries that don't correspond. Duplicated in `confusion_analysis.py:60-61`.
**Expected vs actual:** expected — one scaler fit on train, both sets transformed by it; actual — as-coded rep0 QSVM acc 0.3246 matches `rep0_D_qubo.json` exactly, confirming the shipped numbers inherit the bug. Verifier reproduced independent-fit deviations on the real features CSV.
**Impact:** Every QSVM accuracy in `summary_20260826_133947.csv` (0.2410–0.3608) and the per-class table are suspect until rerun. Classical baselines unaffected (raw columns).
**Fix:** Fit once on train: `scaler = MinMaxScaler((0.05,1.52)).fit(Xtr)` → `transform(Xte)`; same pattern in confusion_analysis. Rerun experiment + confusion.
**Timeline: today; results regenerate in minutes.**

### PSI-001 · Dashboard PSI contradicts its documented formula — [HIGH, runtime-verified] · 2–4h · dashboard regeneration
**Problem:** `pollution_dashboard.py:68-83`. Docstring/HTML say hazard-weighted sub-indices (cigarette ×3); implementation uses raw counts over the max zone's *weighted total* — weights never multiply their own class.
**Repro (verifier-run):** implemented PSI = 25/20/15/7/11/20 for Zones A–F (matches `zone_table.csv`); the documented formula yields 100/100/96/33/75/99. All six zones fall in the 0–50 "Good" band — PSI_BANDS, colors, and the worst-zone KPI are dead UI.
**Impact:** Anyone re-deriving PSI from the printed method gets values 4× different; the cigarette-×3 story (central to the narrative) is absent from the metric.
**Fix:** Decide one formula — recommended: weighted sub-index `s_c = count_c · W_c`, normalized by the max over (zone, class) of `s_c`, ×500 dominant-pollutant rule. Regenerate dashboard + zone_table; update the method note. **Timeline: this week.**

### MET-001 · No significance testing; n=5 underpowered — [HIGH] · 2–4h + ~30min machine time
**Problem:** `run_experiment.py:89-94` reports mean/std only. QUBO−PCA gap (0.96pt) and QUBO−MI gap (2.48pt) cannot be distinguished from seed variance at n=5; the paper's central comparative claim is statistically unsupported.
**Fix:** n=25 paired repeats (seeds 42–66), Wilcoxon signed-rank + paired t with Holm correction, 95% CIs in the summary table. **Timeline: with QML-001 rerun — one combined rerun.**

## Priority P1 — this week

### LKG-002 · Same photo's crops land in both train and test — [MEDIUM, verifier-corrected] · 2h + rerun
**Problem:** `run_experiment.py:48-52` splits crops independently; crops sharing a TACO image (identical lighting/sensor/background) straddle the split. Verifier corrected the auditor's numbers: on the real features CSV, 173 of 459 test crops share an image ID with train — the mechanism and its direction (optimistic bias) hold; magnitude misestimated by the auditor.
**Fix:** `GroupShuffleSplit`/`GroupKFold` on `image_id` (already recoverable from crop filenames); report both grouped and ungrouped numbers if the grouped regime is materially harder. **Depends: QML-001 (rerun once).**

### DOC-001 · README wrong in three ways — [MEDIUM, live-reproduced] · 1–2h
Documented run command crashes (`FileNotFoundError: 'D:\Quobo\--config'` — `run_experiment.py:104` reads argv positionally, no argparse, no `--config` flag exists); phantom `psi.py` module (real: `scripts/pollution_dashboard.py`); stale `FidelityQuantumKernel` (actual: `FidelityStatevectorKernel`, qsvm.py:33); stages listed as pending that are complete. **Fix: one-pass correction + argparse CLI so the documented command becomes true.**

### CCH-001 · Feature cache stale-served — [MEDIUM] · 2–3h
`features.py:56-62` reuses `features_6class.csv` unconditionally; no fingerprint ties it to the crops. Changing crops/margin/filters silently serves old embeddings. Cache key is `len(classes)` only — two different 7-class configs collide. Currently consistent (meta n=1836 == on-disk 1,836 crops, verified), but it is a trap for every future iteration. **Fix: store crop-count + per-class counts + file-list hash + class list in `features_meta.json`; validate before reuse.**

## Priority P2 — next two weeks

### LKG-001 · StandardScaler+PCA fit on full dataset — [MEDIUM] · 4–8h refactor, or 0.5h documented limitation
`features.py:67-72` fits PCA on all 1,836 crops pre-split. Identical across arms (relative ranking survives), but absolute accuracies are optimistic and a strict reviewer flags it. **Fix (proper):** restructure `features.py` to emit raw 1280-d embeddings; fit scaler+PCA inside each split in `run_experiment.py` — this also enables the per-split PCA-refit protocol reviewers expect. Cheap alternative: explicit limitation paragraph in the report (transductive preprocessing, uniform across arms).

### CFG-001 · Six dead config keys; two-YAML drift — [MEDIUM] · 1–2h
`qsvm.feature_map`, `experiment.baselines` and four others are silent no-ops; `experiment.yaml` vs `experiment_6class.yaml` have drifted. **Fix: delete dead keys or wire them; single canonical config; fail-fast on unknown keys.**

### DRY-001 · confusion_analysis.py duplicates the split/select/subsample/train pipeline — [MEDIUM] · 2–4h
Reimplements run_experiment's loop with a *different subsample seed protocol* (qsvm.py:73 fixes seed 42 for all repeats; confusion_analysis.py:55 uses seed+rep), so per-class metrics train on different subsamples than the headline run. **Fix: one shared `run_repeat(arm, rep, cfg)` in src; both entry points call it. Kills the divergence class permanently.**

### SEED-001 · Subsample seed constant across repeats — [LOW] · 0.5h
Same 400 training rows drawn for every repeat in the main run (qsvm.py:73). **Fix: `seed + rep`, bundled with DRY-001.**

### PERF-001 · QUBO SA 5× wasted reads — [MEDIUM, benchmark-verified] · 0.5h
`selection.py:103-105` runs `num_reads=max(50, repeats)` = 50 when config says 10 — 0.54s/call vs 0.11s at 10 reads, ×13 alpha trials ×5 reps. Selection ~10.3s vs achievable ~3s per rep; total wall ~1.5min vs ~1.0min. **Fix: `num_reads=repeats`; verify accuracies unchanged (SA at n=50 is far from the annealing-noise floor).**

### PKG-001 · No pyproject, sys.path hacks, unpinned deps — [MEDIUM] · 2h
Three files copy-paste `sys.path.insert`; `pip install -e .` impossible; requirements unpinned for the entire classical stack (sklearn coupling is strongest — `mutual_info_classif`/`KBinsDiscretizer` behavior drifts across versions; verified: 1.6.1 in venv). **Fix: pyproject.toml (packages=src/quobo, dev extras: pytest/ruff), constraints.txt from `pip freeze`, delete hacks.**

## Priority P3 — backlog

| ID | Sev | Problem | Effort | Fix |
|---|---|---|---|---|
| BBOX-001 | Low (verifier downgraded from High: real-data impact ≈0; code defect real) | Clamp uses original-image dims after rescaling into downloaded-image space (`data_prep.py:156-157`) | 0.25h + 2h test | Clamp to `img.shape`; synthetic-COCO unit test with analytic slice |
| TST-001 | High→(as test debt) | Only test is a script-style smoke test with a near-tautological assertion (`len(sel)==k`); arms A/B/C cannot structurally fail it; `sys.argv` mutation at module level | 3–4h restructure | See §6 |
| ZONE-001 | Low | Non-batch filenames silently → Zone A; documented 'folder' mode unimplemented | 0.5h | Implement `data/geo/<Zone>/` scan or delete the docstring promise |
| MI-BIN | Low | MI at 16 quantile bins on 1,377 rows carries an upward bias floor; MI-vs-QUBO comparison slightly unfair to QUBO | 0.5h note / 2–4h bias study | Note in report; or bias-corrected MI (Miller-Madow) |
| RETRY | Low | Dead Flickr links sleep up to 20s each (4 retries × 2 URLs) | 0.5h | Fail fast on 404/410; keep backoff for timeouts |
| PROV | Low | TACO snapshot date/hash unrecorded anywhere | 1h | Record source URL + sha256 + counts in data_prep_stats.json (survey found v1.0=1,500 vs living dump=3,831 — which snapshot matters) |

### Security posture (all verified; none exploitable in current scope)

| ID | Severity | Issue | Rating basis | Fix |
|---|---|---|---|---|
| SEC-001 | Low | No size cap on downloads; full-body images read into memory | CVSS ~2.7 (local, trusted sources, DoS-only) | Byte counter cap in `download_file`; `len(r.content) ≤ 20MB` guard in fetchers |
| SEC-002 | Low | Filename sanitization trusts dataset ('\\', '..' pass through) | CVSS ~3.1 (path traversal requires malicious annotations.json) | `dest.resolve().is_relative_to(root)` assert; reject '..' |
| SEC-003 | Low | Dashboard HTML built via unescaped f-strings | CVSS 0 *today* (all interpolated values are code-controlled constants — verifier audited every call site); latent if citizen data ever feeds it | `html.escape()` at 3 call sites; comment the invariant |
| SEC-004 | Medium (supply-chain) | Unpinned dependency tree | — | Constraints file (PKG-001) |

No Critical/High security vulnerabilities. The Medium items are methodology/quality risks, not attack surface.

---

# 3. Enhancements & Modifications (per component)

**`src/quobo/qsvm.py`** — (1) fit-once scaler (QML-001); (2) per-repeat subsample seed (SEED-001); (3) return per-class predictions from `run_all_classifiers` so downstream analysis never re-trains (currently `evaluate()` discards them — this is the root cause of DRY-001); (4) optional `DummyClassifier` rows (majority-class baseline is the honest floor: bottle=23.75%, not uniform 16.7% — Random-8 QSVM's 24.1% is *exactly* majority-class, a striking honest result worth reporting); (5) persist `fit_seconds`/`predict_seconds` per arm (already captured — surface them in the summary table; QSVM 0.79s fit vs RBF 0.006s is a real cost worth stating).

**`src/quobo/selection.py`** — (1) num_reads fix (PERF-001); (2) hoist QUBO matrix construction out of the α-bisection loop, update only diagonal per α; (3) record `alpha`, per-α subset size, and SA energies in the rep JSON (stability claims currently have no persisted evidence; features 7,8,1,6 appearing in ≥13/15 runs is a strong story left untold); (4) delete the dead else-branch and stale comment at lines 110–146; (5) optional CMI-QUBO variant (conditional-MI off-diagonal, Fraunhofer 2024) as a low-cost methods extension — the survey found it beats plain MI when information is diffuse.

**`src/quobo/features.py`** — (1) cache fingerprint (CCH-001); (2) emit raw embeddings + move scaler/PCA into the split loop (LKG-001) — this is the single refactor that makes the whole methodology review-proof; (3) `.npz` companion for fast reload; (4) `Image.verify()` sweep over crops (1836 files, seconds) as a data-validation gate.

**`src/quobo/data_prep.py`** — (1) bbox clamp fix (BBOX-001); (2) group annotations by image_id, decode each image once (worst image currently decoded 90×); (3) size caps + path containment (SEC-001/002); (4) provenance record (PROV); (5) per-image memoization also makes the crop stage deterministic against mid-run file changes.

**`src/quobo/run_experiment.py`** — (1) argparse CLI (fixes DOC-001's crash); (2) grouped split option (LKG-002); (3) significance tests in the summary (MET-001); (4) per-class metrics for all classifiers, not just RBF; (5) log to file via logging config, retire the six ad-hoc `*.txt` redirects in the repo root.

**`scripts/pollution_dashboard.py`** — (1) PSI formula fix (PSI-001); (2) predictions-ingestion mode (S1); (3) implement or descope 'folder' mode (ZONE-001); (4) html.escape (SEC-003); (5) Folium choropleth (S6); (6) `zone_of` returns None + explicit "unassigned" category instead of silently blending into Zone A.

**`scripts/confusion_analysis.py`** — collapse into a thin consumer of `run_repeat()` after the DRY-001 refactor; add QSVM per-class rows (currently RBF-only); load selections from the experiment JSONs instead of recomputing ~9s of SA per rep.

---

# 4. Advanced Features (differentiating, ordered by value/effort)

**A1. Uncertainty-aware PSI (methods first, no precedent).** Propagate per-class classifier error (confusion-matrix posterior) into zone-index confidence bands. The survey verified no air-quality index does this; it converts the dashboard from a demo into a citable methodological contribution.

**A2. Kernel-design ablation map.** Sweep reps {1,2,3} × entanglement {linear, full} × feature budget {4,6,8,12} on the fixed selection. Produces the figure reviewers remember — an empirical map of the 8-qubit concentration safe zone — and operationalizes Schuld 2021. If QSVM tracks RBF across the map, that *is* the parity story with evidence.

**A3. Trainable quantum kernel (QKTA).** `TrainableFidelityStatevectorKernel` + SPSA to maximize kernel-target alignment on the QUBO-8 features. Moderate effort; directly attacks the QSVM-vs-RBF gap (34.1 vs 41.4) rather than reporting parity.

**A4. Real QPU runs (both paradigms).** D-Wave Leap for EXP D (S3); IBM Heron for the 8-qubit ZZFeatureMap with SamplerV2 + ZNE. Even one hardware-validated number per paradigm moves the paper a tier.

**A5. Edge deployment benchmark.** INT8 MobileNetV2 + the 8-feature head via ONNX Runtime on CPU; report ms/scan. The survey's resource-constrained literature (WasteNet 97% on Jetson, 95.98% @ 4.7W) gives direct comparables — an angle no quantum-waste paper has.

**A6. Geographic hot-spot enforcement analytics.** Zone rankings with binomial CIs on cigarette-detection rates (the class the system measures best: 72.2% RBF recall) — the Litterati SF cigarette-tax precedent ($4M/yr) is the policy hook; pairs with the GMC briefing.

---

# 5. New Additions (practical, architecture-aligned)

1. **`src/quobo/reporting.py`** — single module generating all tables/figures from experiment JSONs (kills the results-generation duplication scattered across scripts).
2. **LASSO + mRMR arms (E, F)** in `selection.py` — both are ≤20 lines; turns the 4-arm ablation into the 6-arm suite the SIGIR22/Mücke papers use.
3. **Grouped-split protocol** — `image_id` column added to the features CSV at build time.
4. **GeoJSON export** — dashboard emits `zones.geojson` (Ward name, PSI, cleanliness, dominant class, CI) for direct QGIS ingestion — the ESA-SNAP/QGIS story in the plan becomes a real artifact.
5. **CPCB/NCAP AQI correlation stub** — fetch Gwalior station AQI (3–4 stations documented), correlate ward PSI; even a null correlation is reportable and pre-registers the validation pathway.
6. **OpenLitterMap/Debris-Tracker ingestion adapter** — replaces the demo zone mapping with real geolocated observations where available; ODbL-compatible.
7. **GitHub Actions CI** — ubuntu, py3.12, pip install -e .[dev], ruff + pytest on the synthetic-data suite; Qiskit/TensorFlow CPU wheels exist, no GPU needed.
8. **Zenodo release** of the derived crops/features CSVs (TACO data is CC BY 4.0 — attribution required; TrashBox has *no license* — do not redistribute TrashBox-derived artifacts).
9. **`configs/paper.yaml`** — one frozen config (all fixes + n=25) whose hash appears in the paper; ends config-drift risk permanently.

---

# 6. Verification & Testing Strategy

**Layer 0 — data validation gates (new `scripts/validate_data.py`, also a test):**
annotations schema (required keys, unique ann ids, category_id ∈ categories, bbox w/h > 0 — spot-check passed on 2,000 real bboxes); `Image.verify()` sweep over all 1,836 crops; crops ↔ features-CSV count/class consistency (the CCH-001 fingerprint check); TACO snapshot hash recorded.

**Layer 1 — unit tests (pytest, replacing the smoke script):**
- `test_scaler_isolation`: scaler params derived from train only; test values outside [0.05,1.52] map outside (no clip).
- `test_bbox_clamp`: synthetic COCO (200×100 record, 400×200 saved image, bbox [50,25,40,20]) → analytic crop slice with 10% margin.
- `test_psi_hazard_propagation`: zone with 1 cigarette vs 3 bottles → cigarette sub-index 3× the bottle's; band thresholds.
- `test_zone_of`: batch_1→A … batch_6→F, batch_7→A (wrap), batch_11→E, non-batch→unassigned.
- `test_selection_recovers_ground_truth`: synthetic features where 6 of 50 columns carry signal; QUBO/MI must recover ≥5 informative indices — the assertion the smoke test should have been.
- `test_reproducibility`: same seed → identical selection, identical energies, bit-equal accuracy.
- `test_qubo_matrix`: finite values, symmetry, cardinality penalty shape at λ extremes.
- `test_subsample_parity`: main-run and confusion-run produce identical index vectors for the same seed (the DRY/SEED regression guard).

**Layer 2 — integration test (<5 min):** ~120 synthetic crops → MobileNetV2 (random weights, no download) → PCA-20 → 4 arms at reduced sa_sweeps → QSVC on 90 samples → metrics JSON asserted against golden tolerances (accuracy within ±0.02 of a checked-in golden; catches API regressions like the qiskit-ml 0.8→0.9 import break the survey documented).

**Layer 3 — statistical regression harness:** the n=25 run's summary CSV checked in as golden; CI test asserts headline means within ±1.5σ after any change — this is what protects the paper numbers through every refactor above.

**Layer 4 — performance tests:** budget assertions on the measured stages: QUBO selection ≤5s/rep (post PERF-001), QSVM fit ≤1.5s at 400 samples, full experiment ≤4min at n=25. Already-benchmarked baselines recorded so regressions are flagged, not discovered.

**Layer 5 — security review (light, threat-model appropriate):** supply-chain (constraints file hash), path containment on all dataset-derived paths, size caps, escaped HTML. A one-time `pip-audit` run; the surface is small because the project pulls no untrusted input today.

**Layer 6 — user acceptance (the civic demo):** the dashboard's acceptance test is non-technical: a zone-by-zone walkthrough where the PSI ranking, dominant-class bars, and cleanliness score each reconcile with a hand-computed example from zone_table.csv; plus the explicit demo-zone disclaimer check. For the academic audience, the UAT equivalent is the reproducibility pack: fresh-clone → one command → golden numbers within tolerance.

---

# 7. Beyond the categories — sequencing, risk, and publication

**The single most important fact about the fix list:** QML-001, MET-001, LKG-002, SEED-001, PERF-001, and DRY-001 all land on the *same rerun*. Do the code fixes as one batch, then regenerate everything once — never pay the rerun cost per-issue.

**Sequenced plan:**
- **Day 1:** VCS-001, QML-001, PERF-001, DOC-001, PSI-001 → rerun → commit. *(~6h)*
- **Week 1:** DRY-001 refactor (+SEED), CCH-001 fingerprint, BBOX-001, MET-001 at n=25, grouped split LKG-002 → the "paper freeze" rerun → tag `v1.0-results`. *(~2 days)*
- **Week 2:** LKG-001 refactor (raw embeddings + in-split PCA), pytest suite layers 0–2, pyproject/CI. *(~2 days)*
- **Weeks 3–4:** strategic layer — S1 predictions-mode dashboard, S3 D-Wave run, S2/S6 arXiv+map, LASSO/mRMR arms, cross-dataset replication on TrashBox/TrashNet.
- **Then:** publish (FRUCT first — the venue of the nearest prior paper is both thematically apt and citationally strategic), Zenodo release, GMC briefing.

**Honest-results framing (unchanged from the research report):** expect QUBO-8 ≈ MI-8 statistically after 25 repeats unless the correlated-feature effect is real; expect QSVM ≈ RBF-SVM; the defensible claims are methodology, first-of-kind application, statistical discipline most of the literature lacks, and the integrated analytics layer. The audit's central recommendation is to make those claims *bulletproof before making them loudly* — every number in the paper should survive a reviewer rerunning the repo with the standard protocol.

**Risk register (top 3):** (1) *Novelty erosion* — unversioned, unpublished firsts; mitigation: S2 immediately. (2) *Methodological rejection* — the P0/P1 fix list is exactly the review checklist; mitigation: complete before submission. (3) *TACO provenance* — snapshot ambiguity (1,500 vs 3,831 images) and tacodataset.org instability; mitigation: PROV hash now, count-on-download, state the snapshot in the paper.

---

*Every claim above traces to: `research/audit/audit_{bugs,methodology,performance,quality,security,testing}.json` (57 findings with file/line evidence, effort estimates, and fixes), `research/audit/verified_critical.json` (13 adversarial verification verdicts), `research/audit/strategy.json` (16 opportunities), and `research/RESEARCH_REPORT.md` (the underlying 222-finding literature survey).*
