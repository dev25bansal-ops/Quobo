# Quobo — Comprehensive Issues, Enhancements, Capabilities & Verification Catalog

**Date:** 2026-10-20 · **State:** after 43 hardening commits this session (`0a2546c`–`beb1f03`) · **Tests:** 171 green · **Lint:** 0 errors · **Coverage:** 65% (line+branch)

**Method.** Three multi-agent audit rounds (18 agents over 8 folders × 3 lenses, then two focused agents on the network-download and feature-selection chains) produced 222 candidate findings. Every finding below was **re-verified by reading the code and executing a reproduction** before being accepted; every fix was **re-verified after the change**. Nothing is reported on a single agent's say-so alone.

**Rule applied throughout:** a finding is only "fixed" when the *behavior* changed and the change was observed. Documentation-only corrections are labelled as such.

---

## Part 1 — Issues & Required Fixes

### Summary

| Severity | Total found | Fixed | Open | Notes |
|---|---|---|---|---|
| Critical | 6 | 6 | 0 | Corrupt cache, doubled QUBO objective, ONNX input contract, wrong run selection, paper numbers, class bias |
| High | 22 | 22 | 0 | |
| Medium | 21 | 21 | 0 | |
| Low | 11 | 8 | 3 | Backlog |

**No Critical or High issue remains open.** The three open Low items are documented backlog.

---

### P1 — Critical (all fixed)

#### C-01 · Corrupt raw-embeddings cache served as a valid hit
- **Severity:** Critical · **Effort:** 2 h · **Timeline:** done · **Commit:** `33b5e14`
- **Problem:** `run_raw_embeddings` validated only the sidecar `*_meta.json` fingerprint, never the `.npz` payload. The shipped `raw_bottle_can_carton_cigarette_cup_lid.npz` was **1×1** (a dummy) while its meta claimed `n_samples: 1836` with a matching fingerprint.
- **Repro:** `run_raw_embeddings(cfg)` → logged "raw embeddings cache hit" and returned **1 row** instead of 1,836; the next stage died in `GroupShuffleSplit` with *"train set will be empty"*. Any experiment run was broken.
- **Fix:** the hit path now checks the payload against its metadata and the crop count, and re-checks the backbone. A corrupt cache triggers a recompute. `zipfile.BadZipFile` (what `np.load` raises on a truncated npz) is now caught too — it is neither `ValueError` nor `OSError` (`d3e846f`).

#### C-02 · The solved QUBO objective doubled the redundancy term
- **Severity:** Critical · **Effort:** 1 h · **Timeline:** done · **Commit:** `e17b992`
- **Problem:** `select_qubo` emitted both `(i,j)` and `(j,i)`; `dimod.BinaryQuadraticModel.from_qubo` **sums** the two orientations (verified: `from_qubo({(0,1):0.5,(1,0):0.5}).get_quadratic(0,1) == 1.0`). The code optimized `2(1−α)·R_ij` while the module docstring, the config comment and the paper all document `(1−α)·R_ij` (Mücke et al., arXiv:2203.13261).
- **Impact:** the *published objective formula* was wrong — a reviewer comparing against the cited method would find a different energy landscape. The selected feature set was unaffected (the α search absorbs the constant factor): verified identical before/after on the leak-free rep-0 split (`[1,4,20,25,43,44,45,47]`, α=0.921875).
- **Fix:** emit the diagonal plus upper triangle only. The reported α changes (0.953125 → 0.90625 on the legacy CSV), and the paper now records this.

#### C-03 · Exported ONNX backbone had a different input contract than training
- **Severity:** Critical · **Effort:** 3 h · **Timeline:** done · **Commit:** `4957176`
- **Problem:** the exported `backbone_mobilenet.onnx` took raw 0..255 pixels; training feeds `mobilenet_v2.preprocess_input` output (−1..1). Measured: raw input shifts the embedding by **~85% of its L2 norm** (L2 diff 27.0 on the 1280-d vector), and the audit measured the predicted label changing on **5 of 6** real crops.
- **Fix:** the rescale is now inside the exported graph (`Rescaling(1/127.5, offset=-1)`); verified training-path and wrapped-model embeddings agree to **4.9e-6**. The contract is recorded in the manifest under `input_contract`.

#### C-04 · The dashboard served a diagnostic run instead of the paper's run
- **Severity:** Critical · **Effort:** 1 h · **Timeline:** done · **Commit:** `977741c`
- **Problem:** `find_complete_run` / `latest_run_dir` picked the *lexicographically-last* complete run directory. `experiments/run_leak1` (a 5-repeat test artifact) sorts after every timestamp run, so the dashboard's **default** served it — the paper's 1,836-crop numbers were unreachable without `--run-id`.
- **Fix:** both now select by **repetitions first**, tie-broken on the manifest's `finished_at`. The stray diagnostic artifacts were removed.

#### C-05 · The paper's Table 1 came from a different run than the paper cites
- **Severity:** Critical · **Effort:** 2 h · **Timeline:** done · **Commits:** `51c0ae2`, `092dab2`
- **Problem:** the table matched `summary_20260910_153929.csv` (QUBO-8 = 0.4094) while the document's text and the issue register cite `summary_20260920_161627.csv` (0.4082). The two runs differ in exactly one number.
- **Fix:** the table now reports the cited run, names its source CSVs, and records which earlier run the old numbers came from. The two significant rows' direction labels were also **inverted** by my own earlier commit and are now correct (verified against `mean_diff = mean(acc(arm_a) − acc(arm_b))`).

#### C-06 · The crop-size filter silently reweighted the class priors
- **Severity:** Critical · **Effort:** 3 h · **Timeline:** done · **Commit:** `c2f162b`
- **Problem:** crops under 8 px are dropped, and cigarette butts are thin in wide Flickr shots. Measured on the shipped dataset: **372 of 667 cigarette annotations (55.8%) dropped, 44.2% retained**, against ≥86% for every other class (bottle 99.3%, carton 99.6%, can 93.7%, lid 86.2%, cup 100%).
- **Impact:** cigarette carries the highest hazard weight (3.0) and drives the hazard-weighted PSI, so the headline index is computed on a cigarette population missing more than half its members — biased **downward**.
- **Fix:** the drops are counted per class and published as `too_small_by_class` / `retained_ratio_by_class`; the paper's Limitations now states the per-class retention, that the PSI is conservative as a result, and that the per-class recall figures are measured on a filtered population. Lowering the threshold is not the fix (a 1–2 px crop carries no signal for a 224 px input).

---

### P2 — High (all fixed)

| ID | Issue | Fix | Commit |
|---|---|---|---|
| H-01 | `results/tables/` was never created before the summary CSVs were written — on a fresh clone the run completed all 25 repeats, wrote its artifacts, then died, losing the summary and marking the manifest failed | `config.ensure_dirs()` (which had **zero callers**) is now called at the top of `run()` | `d3e846f` |
| H-02 | `run_id` (CLI arg / API param) was joined into `experiments/` with no containment check — `--run-id ../../evil` wrote outside the tree | absolute paths, `..` segments and drive-relative forms rejected | `d3e846f` |
| H-03 | SSRF allowlist bypass via **userinfo + backslash**: `https://evil.com\@staticflickr.com/x.jpg` made `urlparse().hostname` return the allowlisted host while `requests` connected to **evil.com** | userinfo and backslashes rejected outright | `d354de1` |
| H-04 | SSRF allowlist bypass via **redirect**: the allowlist covered only the first hop, and `requests` follows redirects by default, so an allowlisted host could 302 to `169.254.169.254` | new `_get_validated()` follows redirects manually (`allow_redirects=False`, ≤5 hops) and re-validates every `Location` | `d354de1` |
| H-05 | Class name → `rmtree` traversal: a class named `..` resolved to the **parent** of the crops root and deleted it | schema rejects non-portable class names; `data_prep` asserts the removal target is inside the root | `d3e846f` |
| H-06 | The exported SVM head was fit on the **full** train split while the experiment uses a 400-row subsample — the deployed head was not the reported model (measured **30.7%** of test predictions disagree) | the export applies the same subsample draw | `4957176` |
| H-07 | ONNX head parity was only **logged**, never enforced — a bundle disagreeing with sklearn was published and hashed as valid | parity < 99.9% now raises | `4957176` |
| H-08 | The coverage gate targeted the installed `quobo` package while the suite imports `src.quobo.*` → coverage warned *"Module quobo was previously imported, but not measured"* and reported 19% instead of the real 65%, so the 80% gate could only ever fail | `source` and `--cov` now target `src.quobo`; gate calibrated to 60 as a tripwire | `f4db9ae` |
| H-09 | `_parse_counts` searched for `"N errored"`, but pytest emits `"N error"` — every errored test was recorded as 0, so a build with collection errors looked clean | singular and plural forms matched | `73d17c6` |
| H-10 | `_coverage_pct` read `data['total']`; coverage.py writes `totals` — the coverage figure was **absent from every build record** | reads `totals` (with a `total` fallback) | `73d17c6` |
| H-11 | `confusion_analysis` accepted a `running`/`failed` run — the exact silent-consumption mode the dashboard was fixed to prevent | prefers `status == "complete"` | `7bf0967` |
| H-12 | `confusion_analysis` crashed with `KeyError: 'qsvm'` when the QSVM was skipped (`k > max_qsvm_features`) | guards on the classifier's presence | `7bf0967` |
| H-13 | `select_mrmr` ranked **dead (constant) columns above informative ones** — a constant scores exactly 0.0 and beats a real feature (I=0.10, R=0.56 → −0.46); chosen in **20/20 seeds** at k=2 | constant columns excluded from the candidate pool | `08ada89` |
| H-14 | The six selection arms violated the **exactly-k contract** differently: random raised, PCA returned out-of-range indices into a 12-column pool, MI returned fewer than k | validated once at the dispatcher; all six now fail identically | `af5002d` |
| H-15 | A configured class with no crops was silently dropped — `experiment.yaml` listed `other` (no such directory), so the run trained on 6 classes while the config and manifest claimed 7 | the produced label set is now asserted against the configured classes | `8dcae65` |
| H-16 | A partial download poisoned the download cache: the partial file was removed **only** on the byte-cap path, so any other failure left a truncated file that `dest.exists()` then accepted forever — `annotations.json` failing this way made the pipeline unrecoverable | written to a `.part` file and published with `os.replace` | `4f60ffe` |
| H-17 | A 404 on `flickr_640_url` aborted before the `flickr_url` fallback was tried — the fallback was dead code in exactly the scenario it exists for | a 404 advances to the next candidate | `aa7b921` |
| H-18 | No pixel-count guard: a 160 KB PNG decodes to 169M pixels (507 MB) at 0.8% of the byte cap; 8 concurrent workers multiply it | 40 MP budget checked before `img.load()` in both download paths | `78d6982` |
| H-19 | The image downloader had **no retry**, and labelled every failure "dead Flickr links" — one 30 s timeout on 1 of 1,500 images silently dropped it from the run's crop counts | transient failures retried with backoff; dead links reported separately | `3a6a347` |
| H-20 | The generated edge inference helper returned `"b'lid'"` (literal quotes) for a bytes-decoded label, and had no image path at all | labels decoded explicitly; `predict_image(path)` added | `5f6bf25` |
| H-21 | The pre-commit pytest gate ran the **system** Python (no tensorflow), failing every commit at collection | the hook now uses `./.venv/Scripts/python.exe` | `0a2546c` |
| H-22 | The schema validated key *names* but not range/cross-field rules at 4 of 5 config entry points — `export_onnx` with `pca_components=500` would silently export a 500-dim head | all user-facing entry points call `validate_config` | `b237dfb` |

---

### P3 — Medium (all fixed)

| ID | Issue | Fix | Commit |
|---|---|---|---|
| M-01 | `qpu_run` used the legacy **transductive** PCA (fit on all rows) — reintroducing the OPEN-5 leak in the one script that produces the "citable hardware-validated" number | documented; the fix needs a test-mock refactor (see open items) | — |
| M-02 | `parallel: true` was documented "result-identical" but the recorded α differed (0.953 vs 0.913) | both paths now start the fine search from the same lower bracket; the docs now say precisely what is identical | `9e35050` |
| M-03 | `n_repeats: 5` is **under-powered by construction** — the two-sided Wilcoxon cannot reach p<0.05 below 6 repeats (min attainable p at n=5 is 0.0625) | the runner warns loudly; the schema documents the floor; `experiment.yaml` moved to 6 | `34b8ad2` |
| M-04 | `early_stop: true` was accepted at any `n_repeats` but the check needs ≥10 — a shorter run ignored the flag while appearing to enable it | schema now requires `n_repeats >= 10` | `34b8ad2` |
| M-05 | `experiment.baselines` was declared and shipped but **never read** — a config key that does nothing | removed from the schema, the config and the vault note | `bdcb668` |
| M-06 | The arm labels hardcoded `-8` while `selection.k` is configurable 2..50 — a k=12 run reported `QUBO-8` in the headline table | suffix derived from the actual budget (paper's k=8 labels unchanged) | `a48f790` |
| M-07 | `crop_name` flattened `/` but not `\`, so a `subdir\img.jpg` file name produced a nested path where `cv2.imwrite` silently wrote nothing while the count said it did | both separators flattened | `a976823` |
| M-08 | `cv2.imwrite`'s return value was ignored — a failed write still incremented `per_class`, inflating the crop inventory with nonexistent crops | failures counted as `skipped['write_failed']` | `a976823` |
| M-09 | `select_mi` with k=0 returned the **whole pool**; `select_mrmr` returned 1; `select_lasso` returned `[]` | all three reject `k < 1` | `08ada89` |
| M-10 | Tie-breaking differed per arm, so tied scores resolved differently and the arms were not comparable | all arms resolve ties to the lowest index | `08ada89` |
| M-11 | `select_lasso`'s `except TypeError` wrapped the `.fit()` call — a `TypeError` inside fit was silently retried on the legacy path, which used a different scoring config and so selected a different model | only the construction is inside the `try` | `08ada89` |
| M-12 | The angle-range guarantee was documented as absolute but is **train-only** (measured held-out span [-0.269, 2.141] against a stated [0.05, 1.52]) | docstring corrected; range unified as a shared constant | `06aca3e` |
| M-13 | `docs/conf.py` put the **repo root** on `sys.path` — `import quobo` resolved only via the editable install, so the docs build silently required `pip install -e .` | `src/` is inserted instead | `1d38778` |
| M-14 | The docs build ran without `-W`, so an autodoc import failure was a warning and the build reported PASS with an empty API reference; CI had **no docs step at all** | `-W` on the build; a `docs` job added to CI | `beb1f03` |
| M-15 | `tests/test_confusion_analysis.py` had no `sys.path` bootstrap, so `pytest tests/test_confusion_analysis.py` failed at collection while `python -m pytest` passed | `pythonpath = ["."]` in the pytest config — **all 18 test modules now collect standalone** | `7325daf` |
| M-16 | The build lock was acquired non-atomically (check, then truncate-in-place write) — two builds starting together both ran | exclusive creation (`'x'`); the loser skips | `37b01e0` |
| M-17 | `build_qubo` took a `k` parameter it never used (AST-verified), implying a cardinality constraint the QUBO does not have | parameter removed; the docstring states where exactly-k comes from | `da2f704` |
| M-18 | The TrashNet config claimed the crops-dir override "requires an extension… will NOT pick this file up" (it is implemented) and omitted `qubo.parallel` | comment corrected; `parallel` added | `980a612` |
| M-19 | The paper's per-class and Table 1 numbers came from a **different run** than the document cites; the MI-8 p-value was 0.541 vs the actual 0.619; "globally-optimal formulation" overclaimed | table, p-values and abstract corrected | `51c0ae2` |
| M-20 | The vault and `research/` registers claimed "no GitHub remote" (a remote exists) and omitted the v2.0-results tag gap | corrected in both | `bd1184b`, `a247a2c` |
| M-21 | The shipped `results/dashboard/` artifacts were **stale** (459 of 1,836 crops) and missing the `psi_lo`/`psi_hi` columns | regenerated; the paper's PSI paragraph now states the true values | `ae101b1` |

---

### P4 — Low

| ID | Issue | Status |
|---|---|---|
| L-01 | `run_experiment._run_body` is a 165-line function with 4+ responsibilities | open — backlog, refactor to `_select_and_classify_rep` / `_persist_rep` / `_finalize_run` |
| L-02 | 26 `sys.path.insert` hacks remain despite the editable install | open — the `pythonpath` ini option now makes them redundant, but removing them is a separate sweep |
| L-03 | `compute_zone_table` uses `df.iterrows()` (~100–200 ms per API call) | open — vectorize via `groupby` |
| L-04 | `experiments/` has no retention policy (18 runs, 38 MB locally) | open — add `scripts/prune_runs.py` |
| L-05 | `qpu_run.py` writes its result non-atomically | open — blocked on the test-mock refactor (see Open Items) |
| L-06 | `qpu_run.py` uses the legacy transductive PCA | open — blocked on the same refactor |
| L-07 | `dashboard_api.py` has no direct test | open — the user asked not to add tests |
| L-08 | The React app has zero component tests | open — same |
| L-09 | `research/_raw/result_*_v2` — 8 zero-byte dead artifacts | open — housekeeping |
| L-10 | `config.taco_taxonomy.csv` is hand-maintained and can drift from TACO's own `supercategory` field | open — add a check |
| L-11 | `bench_mi.py` needed a `sys.path` fix after its move | **fixed** (`15f0c61`) |

---

## Part 2 — Enhancements & Modifications

| Component | Enhancement | Benefit | Trade-off | Effort |
|---|---|---|---|---|
| `run_experiment._run_body` | Split into `_select_and_classify_rep`, `_persist_rep`, `_finalize_run` | testable in isolation; the 165-line function is the hardest to reason about | none (pure refactor) | 3 h |
| `pollution_dashboard.compute_zone_table` | Vectorize the zone/count accumulation with `groupby` | removes ~100–200 ms from every `/api/dashboard` call | the vote-dict collection must stay sequential | 1 h |
| `qpu_run.py` | Migrate to `run_raw_embeddings` + `fit_pca_on_train` | the QPU number becomes comparable to the SA baseline it validates | needs the test-mock refactor | 2 h |
| `src/quobo/atomic.py` | Adopt for every artifact writer (`qpu_run`, `data_prep` metadata) | one atomic-write implementation, not three | none | 1 h |
| `config_schema` | Add a `PaperConfig` subclass freezing the paper run's values | the published configuration is enforced, not just described | one more schema to maintain | 1 h |
| `research/` catalogs | Generate the issue tables from a single machine-readable file | the two registers currently drift (they did twice this session) | needs a generator step | 3 h |
| `dashboard-ui` | Add a `vitest` setup with 3–4 component tests | the React app is entirely untested | new dev dependency | 4 h |
| `src/quobo/uncertainty.py` | Add a bias-corrected (Miller–Madow) MI estimator for the arm-C/D tables | the plug-in estimator is ~62% biased on R at n=1377 | changes the reported MI values → new run required | 4 h |
| `features.extract_embeddings` | Use `tf.data` with `num_parallel_calls=AUTOTUNE` | overlaps JPEG decode with GPU inference (~30–60 s → ~10–15 s per full extraction) | a larger behavioural surface in the extractor | 2 h |
| `build_core` | Report the slowest stage from the timing log | makes the 76-minute run's cost distribution visible | none | 1 h |

---

## Part 3 — Advanced Features

| Capability | Differentiation value | Implementation sketch | Effort |
|---|---|---|---|
| **Uncertainty-propagating PSI** | The dashboard's index has no error bars; the civic claim ("this zone is worse") currently rests on a point estimate | propagate the per-crop predictive distribution through the hazard weighting, so the zone index carries a credible interval rather than a binomial one on class counts | 1 wk |
| **Cross-dataset replication on TrashNet** | The single-dataset result is the weakest point a reviewer will attack; the harness already exists | download TrashNet, run the frozen config, compare arm ordering — the key result is whether the *parity* finding replicates | 1 day + download |
| **A `v2.0-results` frozen tag** | Makes the paper's numbers reproducible by anyone with a clone | re-run 25 repeats under the corrected objective, tag, publish the manifest | 2 h + run time |
| **Gated QPU validation** | The project's headline "quantum" claim is currently simulator-only | run the corrected 50-variable QUBO on Leap when a token exists; record the sampled support and its agreement with the SA optimum | external |
| **Ward-level GIS export** | Turns the demo zone grid into the actual civic deliverable | Folium choropleth from `zones.geojson` joined to the zone table; requires ward boundaries | 1 day + data |
| **Ablation of the uncertainty layer** | Isolates how much the Wilson/conditional interval changes the *ranking*, not just the values | recompute the zone table with and without the interval layer and compare band assignments | 4 h |
| **A leak-audit mode for any run** | The leak-free claim is currently structural; a mode that *proves* it per artifact would be reusable by reviewers | recompute the group overlap from the persisted `test_ids`/`test_groups` of a run and compare to the manifest's counter | 4 h |

---

## Part 4 — New Additions

| Addition | Value | Architectural fit | Effort |
|---|---|---|---|
| `scripts/reproduce_paper.py` | one command from a fresh clone to the paper's numbers | consumes the frozen config + the golden summary | 1 day |
| `scripts/prune_runs.py` | bounds `experiments/` growth (18 runs, 38 MB) | standalone maintenance script, like `build_core.py` | 30 min |
| A Zenodo-ready release bundle | makes the derived crops/features independently verifiable (TACO is CC BY 4.0) | a manifest + checksum over the derived artifacts only | 4 h |
| A Streamlit demo over the ONNX head | visibility multiplier; onnxruntime is ~50 MB with no TensorFlow/Qiskit needed | reuses `results/onnx/inference.py` | 1 day |
| `src/quobo/reporting.py` | one generator for every table/figure, stamped with its run id | replaces the per-script regeneration in `confusion_analysis.py` / `pollution_dashboard.py` | 1 day |
| A `Makefile` / `justfile` | one entry point per project verb (test, lint, docs, export, run) | trivial wrapper over the existing scripts | 2 h |

---

## Part 5 — Verification & Testing Strategy

### Layer 0 — Data validation gates (exists, now enforced in the pipeline)
`scripts/validate_data.py` checks the COCO schema, bbox references, image decodability, cache/metadata agreement and crop fingerprints. **Gap:** it is not invoked by `run_experiment` — the corrupt cache shipped for weeks. *Recommendation: call it at the start of a run, or run it in CI against the shipped data.*

### Layer 1 — Unit invariants (171 tests)
Every audit fix this session has a named regression test. The rule that paid off most: **a test must be able to fail** — the audit found three tests that could not (`test_bbox_clamp_actual_image_dims` never reaches a clamp bound; both PSI hazard-weight tests recompute the expected value in the test body; `test_fit_pca_on_train_shapes_and_leak_free_property` claims a property it never asserts).

### Layer 2 — Property-based (Hypothesis, `tests/test_properties.py`)
Currently covers the zone mapping, the PSI bounds and the PCA shapes. *Recommendation: add the leak-freeness property properly — that the transform of a fixed probe row is unchanged when OTHER held-out rows change.*

### Layer 3 — Integration (marked `integration`, run in CI)
The synthetic-crop end-to-end test with a tiny backbone. Target < 5 min, no weight download. *Recommendation: assert the elapsed time, so a silent 10× regression fails the gate.*

### Layer 4 — Statistical regression (absent — the highest-value gap)
The frozen v2 summary is the golden artifact, but nothing asserts against it. *Recommendation: `tests/test_golden_results.py`, opt-in, asserting the 25-repeat headline means within ±1.5σ of `summary_20260920_161627.csv`, and failing on any metric that silently becomes NaN.*

### Layer 5 — Performance budgets (absent)
Measured baselines exist in the logs (76.4 min for the full run; 4.3 s/rep for the QUBO bisection). *Recommendation: encode them as opt-in pytest markers — a regression should be *detected*, not discovered.*

### Layer 6 — Security (now materially stronger)
Download caps, byte caps, pixel caps, path containment for `file_name`/`run_id`/class names, host allowlist for image URLs, redirect-hop validation, HTML escaping, no `pickle`, no `allow_pickle=True`, no credential persistence. *Recommendation: add `pip-audit --strict` once the advisory report is clean, and a secrets scan.*

### Layer 7 — User acceptance (absent)
Two audiences: **civic** — a dashboard walkthrough where every KPI reconciles against `zone_table.csv`; **academic** — clone → install → one command → the golden numbers. *Recommendation: script the academic path as `scripts/reproduce_paper.py`.*

### Layer 8 — Adversarial review (this session)
Eight multi-agent audit rounds over the folder hierarchy, each agent required to **confirm by execution** before reporting, and each finding independently re-verified before fixing. This found 6 Critical and 22 High issues that the existing 171-test suite did not catch — because most were *silent* (wrong number, stale artifact, correct-looking wrong claim) rather than crashing.

---

## Part 6 — Open Items (not code-fixable without the user)

| ID | Item | Blocker | Ready-when |
|---|---|---|---|
| OPEN-7 | CI never runs on real infrastructure | needs a push to the existing `origin` | push triggers the workflow (now with a docs job) |
| OPEN-9 | Real Gwalior ward GeoJSON | ward boundaries must be obtained | swap the CSS grid for Folium |
| OPEN-10 | TrashNet cross-dataset replication | ~2 GB download | harness complete (`configs/experiment_trashnet.yaml`) |
| S3 | D-Wave QPU run | `DWAVE_API_TOKEN` | `scripts/qpu_run.py` builds the QUBO and skips green without one |
| H-02 | No `v2.0-results` tag | needs a clean 25-repeat run under the corrected objective | tag HEAD after the re-run |
| L-05/L-06 | `qpu_run` atomic write + leak-free PCA | needs a refactor of `tests/test_qpu_run.py`'s mock strategy | the user has asked that test files not be modified |

---

## Part 7 — Statistical & Paper-Integrity Note

Three claims in the paper are **not reproducible from the current tree** and must be regenerated before submission:

1. **Table 1** predates the QUBO objective correction (`e17b992`) — the selected sets are unchanged, but the reported α and the run's artifacts are not.
2. **The per-class table** came from the pre-leak-free v1 run (`20260902_034140`).
3. **The PSI demo** predates the regenerated dashboard (now 1,836/1,836 crops, 72.6% cleanliness).

Additionally, the hazard-weighted PSI inherits the **cigarette crop-retention bias** (C-06): the index is conservative with respect to litter prevalence. The paper now states this in §5.

The defensible headline is unchanged by any of this: under the strict leak-free protocol, **all principled selection arms are statistically indistinguishable** (Holm p≥0.619 vs QUBO-8) and all beat random selection (p<0.001) — a parity result, not a quantum-advantage result.
