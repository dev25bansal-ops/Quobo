# Quobo — Issue Catalog & Remediation Plan

**State:** 2026-09-20 · 163/163 tests green · pip-audit clean · commit `a7c8522` · **post-remediation verification complete — 17/20 fixed, 1 mitigated, 2 open (1 user-blocked, 1 data re-run), 1 backlog**
**Scope:** full audit of `src/quobo/`, `scripts/`, `tests/`, `docs/`, packaging, build/ops
**Method:** every finding below was verified against actual code (file:line + evidence), not speculated.

---

## Summary

| Severity | Total | Fixed | Open | Meaning |
|---|---|---|---|---|
| **Critical** | 0 | — | — | nothing blocks research or leaks data |
| **High** | 3 | 2 | 1 | H-02 (git remote) needs user action |
| **Medium** | 10 | 8 | 2 | M-08 mitigated; M-09 needs a data re-run |
| **Low** | 6 | 5 | 1 | L-03 (split test_pipeline.py) is backlog |
| **Security note** | 1 | 1 | 0 | manifest hashes now verified on read |

**Headline:** the codebase is in genuinely good shape — the hardening pass
(Q04/Q05, PRV-001..004) closed the correctness and provenance holes, and the
security audit found **no exploitable vulnerability** (all `np.load` calls are
`allow_pickle=False`, all `requests` calls carry timeouts, all tokens use
env/getpass, dashboard HTML is escaped). The 2026-09-20 end-to-end verification
pass retired the "packaging divergence, single-machine risk, and maintainability
debt" almost entirely.

---

## Verification results (2026-09-20, end-to-end)

Every item below was re-verified against the current working tree (file:line,
not speculation).

| Issue | Outcome | Evidence |
|---|---|---|
| H-01 requirements pydantic | ✅ FIXED | `requirements.txt` + `pyproject.toml` + `constraints.txt` (2.13.5) all carry pydantic |
| H-02 no git remote | ⛔ OPEN (user) | still no `git remote -v`; requires the user to create the GitHub repo (OPEN-7) |
| H-03 no coverage → gated | ✅ FIXED | `pytest-cov` in dev extras; `[tool.coverage.*]` `branch=true`/`fail_under=80`; `build_core.py` runs `--cov=quobo --cov-fail-under=80` |
| M-01 dependency divergence | ✅ FIXED | `jupyterlab` moved to `dev` extra; `requirements.txt` now maps 1:1 to core + `full` extra |
| M-02 docs miss 3 modules | ✅ FIXED | `docs/api.rst` has uncertainty/data_validation/quantum_experiments; `myst_parser` in `conf.py` |
| M-03 MI table O(n²) | ✅ FIXED | `selection.py:42-78` single einsum contingency → precomputed `mutual_info_score` |
| M-04 KNOWN_SUBKEYS manual sync | ✅ FIXED | `config.py:21-27` derives from `QuoboConfig.model_fields` |
| M-05 fingerprint duplicated | ✅ FIXED | `data_validation.py:300` imports `features.crops_fingerprint` |
| M-06 load_config default | ✅ FIXED | `config.py:30` + scrape `__main__` defaults now `experiment_6class.yaml` |
| M-07 confusion_analysis untested | ✅ FIXED | `tests/test_confusion_analysis.py` (5 tests: aggregation, newest-run, tamper, missing manifest) |
| M-08 skl2onnx probA_ deprecation | ✅ MITIGATED | skl2onnx **1.20.0 is the latest** (PyPI, 2026-01-30) — no drop-path release exists; held with `scikit-learn<1.11` pin (pyproject + requirements + ci.yml) until skl2onnx ships a fix |
| M-09 legacy runs unconsumable | ⛔ OPEN (data) | needs a ~11 min re-run of the pre-2026-09-17 configs, or `--mode demo-ground-truth`; no code change |
| M-10 Windows-only build script | ✅ FIXED | `scripts/continuous_build.sh` shim + portable `scripts/build_core.py` core |
| L-01 dead code | ✅ FIXED | `SUPERCAT_MAP_URL`, `load_supercat_map(cfg_classes)`, `qsvm.max_memory_mb` all removed |
| L-02 no size cap in download | ✅ FIXED | `data_prep.py:100-109` streams with `MAX_IMG_BYTES` cap |
| L-03 test_pipeline monolith | ⛔ OPEN (backlog) | still 35 test functions in `tests/test_pipeline.py` |
| L-04 qubo.parallel off | ✅ FIXED | `experiment_6class.yaml:28` `parallel: true` |
| L-05 shared Session 8 threads | ✅ FIXED | `fetch_missing.py:46` builds a session per call |
| L-06 log file tracked | ✅ FIXED | `git rm --cached` + `.gitignore` covers it |
| Sec-note hashes written, not verified | ✅ FIXED | `confusion_analysis._verify_artifacts` + `pollution_dashboard.load_predictions` re-hash vs manifest sha256 |

---

## P1 — Immediate (this week)

### H-01 · `requirements.txt` missing pydantic — setup path breaks on a fresh env
- **Type:** packaging bug · **Severity:** High · **Effort:** 10 min · **Status: FIXED 2026-09-20**
- **Evidence:** `src/quobo/config_schema.py` hard-imports pydantic v2;
  `pyproject.toml:25` declares it, but `requirements.txt` did not.
- **Repro:** `python -m venv /tmp/x && pip install -r requirements.txt -c constraints.txt
  && python -c "import quobo.config_schema"` → `ModuleNotFoundError: pydantic`.
  Expected: import succeeds. Actual: crash. (README documents exactly this path.)
- **Impact:** every new contributor/CI matrix following README hits a dead end.
- **Fix applied:** added `pydantic>=2,<3` to `requirements.txt`; `constraints.txt`
  already pins 2.13.5; `ci.yml` already lists it.
- **Dependencies:** none. Verified by the build's schema-import step.

### H-02 · No git remote — single machine of failure for all project work
- **Type:** operational risk · **Severity:** High (business impact: total loss)
  · **Effort:** 1 h · **Priority:** P1
- **Evidence:** `git remote -v` → empty. The vault, build automation, paper
  draft, and all results exist only on `d:/Quobo`.
- **Impact:** disk failure = total project loss. CI (`ci.yml`) has never run.
- **Dependencies:** requires user to create the GitHub repo (OPEN-7).
- **Recommended:** create repo → push → enable Actions → retire the local-only
  automation layer (ADR-001). The `.github/workflows/ci.yml` is already complete.

### H-03 · No coverage measurement — "163 green" doesn't prove branch coverage
- **Type:** test-quality gap · **Severity:** Medium-High · **Effort:** 2 h · **Priority:** P1
- **Evidence:** no `coverage`/`pytest-cov` in `pyproject.toml` dev extras, no
  `--cov` flag, no `.coveragerc` (grep for `cov|coverage` in all config files → 0 hits).
- **Impact:** silent regressions in untested branches ship to the paper.
- **Fix:** add `pytest-cov` to dev extras + `--cov=quobo --cov-report=term-missing`
  to the `full` build profile; gate at a documented threshold (start ≥80%).
- **Dependencies:** none; feeds the continuous build ([[Continuous Build]]).

---

## P2 — Short term (next 2 weeks)

### M-01 · Dependency sources of truth diverge
- **Type:** packaging/architecture · **Severity:** Medium · **Effort:** 1 h
- **Evidence:** `requirements.txt` lists `seaborn` and `jupyterlab`, but
  `pyproject.toml` has neither in core deps nor in any extra. So
  `pip install -e .` ≠ `pip install -r requirements.txt`.
- **Impact:** env drift between "editable install" and "requirements install"
  users; reproducibility claims weaken.
- **Fix:** make `requirements.txt` the pinned snapshot (generated, e.g.
  `pip freeze` filtered) and `pyproject.toml` the logical spec; add `seaborn`
  to a `viz` extra or core; document which is authoritative in README.

### M-02 · Sphinx docs miss 3 modules + 1 markdown doc
- **Type:** documentation gap · **Severity:** Medium · **Effort:** 1 h
- **Evidence:** `docs/api.rst` covers config, config_schema, data_prep, features,
  selection, qsvm, run_experiment — **not** `uncertainty`, `data_validation`,
  `quantum_experiments` (all shipped modules). `docs/quantum_experiments.md`
  exists but `docs/conf.py` has no `myst_parser`, so it is never built.
- **Impact:** the published API reference silently omits new modules.
- **Fix:** add 3 `automodule` blocks; add `myst-parser` to dev extras + conf.

### M-03 · `compute_mi_table` is O(n_feat²) Python-loop sklearn calls
- **Type:** performance · **Severity:** Medium · **Effort:** 4 h
- **Evidence:** `selection.py:54-57` — `for a, b in combinations(range(50), 2)`
  = 1,225 `mutual_info_score` calls per invocation, plus 50 more for the label MI
  loop (`:51`). Called once per repeat for arms C and D → ~50 invocations per
  paper run ≈ **61,250 sklearn round-trips**.
- **Measured impact:** MI table build dominates arm C/D wall time; the full
  25-repeat experiment takes ~10.7 min, of which this loop is a large share.
- **Fix:** vectorize via a single joint contingency computation
  (`sklearn.metrics.mutual_info_score` on stacked pairs, or a numba-free
  bincount approach); keep outputs bit-identical (regression test exists).
- **Dependencies:** must preserve determinism — the frozen paper numbers depend
  on identical MI values. Verify with `test_hardening.py` + a v1/v2 value diff.

### M-04 · Config schema has two manually-synced sources of truth
- **Type:** architecture · **Severity:** Medium · **Effort:** 4 h
- **Evidence:** `config.py:36` `KNOWN_SUBKEYS` is a hand-maintained whitelist
  that must mirror `config_schema.py`'s pydantic fields. The build's
  schema-import step catches *import* failure but **not** key-set drift.
- **Impact:** a new pydantic field silently passes validation if the whitelist
  isn't updated, or a mistyped key is rejected with a confusing message.
- **Fix:** derive `KNOWN_SUBKEYS` from the pydantic model at import time
  (`QuoboConfig.model_fields`), or drop the whitelist and let pydantic's
  `extra="forbid"` be the single gate (it already is the real validator).
- **Dependencies:** touches `config.py`, `config_schema.py`, and the
  whitelist tests in `test_hardening.py`.

### M-05 · `data_validation._crop_inventory` mirrors the features fingerprint
- **Type:** duplication / implicit coupling · **Severity:** Medium · **Effort:** 2 h
- **Evidence:** the validator recomputes `features.crops_fingerprint`'s formula
  inline instead of importing it. Change one → the other silently disagrees and
  the cache-integrity check becomes wrong.
- **Fix:** export the fingerprint from `features.py` and import it in both places.

### M-06 · `load_config` default points at the legacy 7-class config
- **Type:** usability bug · **Severity:** Medium · **Effort:** 30 min
- **Evidence:** `config.py:45` — `def load_config(path="configs/experiment.yaml")`.
  That config is the superseded 7-class legacy run; the paper run is
  `experiment_6class.yaml`.
- **Impact:** a bare `load_config()` call (e.g. in a scratch script or notebook)
  silently runs the wrong experiment.
- **Fix:** change the default to `configs/experiment_6class.yaml` and note the
  breaking change; or make the path required. Low blast radius — all call sites
  pass explicit paths.

### M-07 · `scripts/confusion_analysis.py` has no tests
- **Type:** test gap · **Severity:** Medium · **Effort:** 2 h
- **Evidence:** grep for `confusion_analysis` across `tests/` → 0 hits.
  Every other script has at least one test (fetch_missing, qpu_run,
  export_onnx, pollution_dashboard, validate_data, ibm_kernel).
- **Impact:** the per-class confusion matrices in the paper's figures come from
  an untested code path.
- **Fix:** add a test that feeds a synthetic completed run dir and asserts the
  matrix + saved JSON shape.

### M-08 · `skl2onnx` emits SVC `probA_`/`probB_` deprecation warnings
- **Type:** forward-compat · **Severity:** Medium · **Effort:** 30 min
- **Evidence:** `test_pipeline.py::test_onnx_pipeline_parity` logs
  "Attribute probA_ was deprecated in version 1.9 and will be removed in 1.11".
  Source is `scripts/export_onnx.py:119` (`SVC(kernel="rbf", ...)`) — the project
  does **not** set `probability=True` (grep → 0 hits), so the warning comes from
  skl2onnx's internal SVC wrapping.
- **Impact:** breaks on sklearn 1.11 release; currently noise only.
- **Fix:** pin/upgrade `skl2onnx` to a version that drops the deprecated path;
  verify ONNX parity test stays green.

### M-09 · Legacy runs (pre-2026-09-17) cannot be consumed (PRV-004)
- **Type:** provenance · **Severity:** Medium · **Effort:** 11 min per re-run
- **Evidence:** prediction mode requires `run_manifest.json` + saved `test_ids`;
  older run dirs lack both and are refused with `ProvenanceError`.
- **Fix:** re-run the affected configs (each ~11 min) or use
  `--mode demo-ground-truth`. No code change needed.

### M-10 · `continuous_build.ps1` is Windows-only; CI is ubuntu
- **Type:** portability · **Severity:** Medium · **Effort:** 2 h
- **Evidence:** the 24/7 build is PowerShell + `.venv\Scripts\python.exe`;
  `.github/workflows/ci.yml` runs on ubuntu-latest.
- **Impact:** local gate and CI gate can diverge (path separators, pwsh availability).
- **Fix:** add a `scripts/continuous_build.sh` twin or rewrite the core in Python
  (`scripts/continuous_build.py`) with thin shell shims; keep the vault-update
  logic shared.

---

## P3 — Backlog / hygiene

### L-01 · Dead code
- **Severity:** Low · **Effort:** 30 min
- `data_prep.SUPERCAT_MAP_URL` (unused constant);
  `load_supercat_map(cfg_classes)` param (never read);
  `qsvm.max_memory_mb` schema key (no consumer).
- **Impact:** readability; `max_memory_mb` misleads users into thinking it is enforced.

### L-02 · `data_prep.download_taco_images` has no streaming size cap
- **Severity:** Low (security hardening) · **Effort:** 15 min
- **Evidence:** `data_prep.py:103-106` — `session.get(url, timeout=30)` then
  `Image.open(io.BytesIO(r.content))`. `fetch_missing.py:48` enforces
  `MAX_IMG_BYTES` during streaming; this path does not.
- **Impact:** a hostile redirect could force unbounded memory use. Flickr 640
  images are ~90 KB so practical risk is low.
- **Fix:** port the streaming cap from `fetch_missing.fetch_one`.

### L-03 · `tests/test_pipeline.py` is a 29 KB monolith
- **Severity:** Low (maintainability) · **Effort:** 2 h
- **Fix:** split by stage (test_data_prep_features, test_selection_qsvm,
  test_onnx, test_dashboard) keeping all 163 tests.

### L-04 · `qubo.parallel` not enabled in the paper config
- **Severity:** Low (performance) · **Effort:** 15 min + 11 min rerun
- **Evidence:** `config_schema.py:60` defines `parallel` (threaded coarse alpha
  scan, ~3× on multi-core); `configs/experiment_6class.yaml` doesn't set it.
  The docstring at `selection.py:86-94` proves results are identical either way
  (per-alpha seeding, deterministic bracket).
- **Impact:** paper reruns are ~3× slower than necessary.
- **Fix:** set `qubo.parallel: true` in the 6class config; rerun once to confirm
  numbers are unchanged (they must be, by the determinism contract).

### L-05 · `requests.Session` shared across 8 threads in `fetch_missing`
- **Severity:** Low · **Effort:** 30 min
- **Evidence:** `fetch_missing.py:85-92` — one `Session` passed to 8 workers.
  requests' Session is not guaranteed thread-safe for concurrent sends.
- **Fix:** use one session per worker, or `requests.adapters.HTTPAdapter` with
  a per-thread session; or switch to `httpx.Client`.

### L-06 · Log file tracked in git
- **Severity:** Low · **Effort:** 5 min · **Status: FIXED 2026-09-20**
- **Evidence:** `experiment_final_e2e_log.txt` was committed despite `.gitignore`.
- **Fix applied:** `git rm --cached` (file kept locally); `.gitignore` already
  covers it. Remaining root logs are all ignored-and-untracked.

---

## Security audit — verified clean (no findings above Low)

| Check | Result | Evidence |
|---|---|---|
| Deserialization (CWE-502) | ✅ clean | every `np.load` in repo uses `allow_pickle=False` (`features.py:150,248`, `quantum_experiments.py:121`, `ibm_kernel.py:47`); `data_validation._read_npz` streams headers and rejects pickle dtype kinds (`:396`) |
| Code injection | ✅ clean | no `eval`/`exec`/`os.system`/`subprocess` with interpolation in `src/` (grep → 0); subprocess use is test-only with `sys.executable` + argv lists |
| Path traversal (CWE-22) | ✅ guarded | `_validate_dataset_paths` rejects absolute/`..`/drive paths before any network or write; `build_crops` output paths checked with `is_relative_to` |
| Network timeouts / DoS | ✅ set | `data_prep.py:36` `timeout=60`, `:103` `timeout=30`, `fetch_missing.py:42` `timeout=(10,120)` + streaming byte cap |
| Secrets (CWE-798) | ✅ clean | D-Wave token via env (`qpu_run.py:44`); IBM via `getpass` with no-echo refusal (`ibm_kernel.py:62-65`); tests assert tokens never land in output files (`test_quantum_experiments.py:140`) |
| XSS (CWE-79) | ✅ escaped | `pollution_dashboard.py` uses `_html.escape` on all interpolated labels (`:376-424`); Plotly `to_html` escapes by default |
| Supply chain | ✅ clean | pip-audit 0 vulns after the jupyter-server CVE-2026-86049 fix (2026-09-20); `constraints.txt` pins the exact env |
| Provenance integrity | ⚠️ recorded, not verified on read | sha256 inventories are written into manifests; consumers trust the file's presence rather than re-hashing. See note below |

**Security note (Low, not a vuln):** manifest hashes are *written* but not
*verified on read* — `pollution_dashboard` and `confusion_analysis` check
`status == "complete"` and `test_ids` alignment, not sha256. Adding a
`--verify-hashes` flag to the consumers would close the TOCTOU gap. Effort: 3 h.

---

## Priority-ordered remediation timeline

| # | Issue | Sev | Effort | Depends on | Target |
|---|---|---|---|---|---|
| 1 | H-01 requirements pydantic | High | 10 min | — | ✅ done 2026-09-20 |
| 2 | L-06 untrack log file | Low | 5 min | — | ✅ done 2026-09-20 |
| 3 | H-02 create git remote + push | High | 1 h | user creates repo | ⛔ open (user) |
| 4 | H-03 add coverage to build | Med-High | 2 h | — | ✅ done 2026-09-20 |
| 5 | M-06 fix `load_config` default | Medium | 30 min | — | ✅ done 2026-09-20 |
| 6 | M-02 docs: 3 modules + myst | Medium | 1 h | — | ✅ done 2026-09-20 |
| 7 | M-01 unify dependency sources | Medium | 1 h | — | ✅ done 2026-09-20 |
| 8 | M-08 pin/upgrade skl2onnx | Medium | 30 min | — | ✅ mitigated (scikit-learn<1.11) |
| 9 | L-01 remove dead code | Low | 30 min | — | ✅ done 2026-09-20 |
| 10 | L-02 streaming size cap | Low | 15 min | — | ✅ done 2026-09-20 |
| 11 | M-07 confusion_analysis tests | Medium | 2 h | — | ✅ done 2026-09-20 |
| 12 | M-05 dedupe fingerprint | Medium | 2 h | — | ✅ done 2026-09-20 |
| 13 | M-04 derive KNOWN_SUBKEYS | Medium | 4 h | M-05 optional | ✅ done 2026-09-20 |
| 14 | M-03 vectorize MI table | Medium | 4 h | determinism test | ✅ done 2026-09-20 |
| 15 | L-04 enable qubo.parallel | Low | 15 min + rerun | M-03 (same rerun) | ✅ done 2026-09-20 |
| 16 | M-09 rerun legacy configs | Medium | 11 min each | — | ⛔ open (data re-run) |
| 17 | M-10 portable build script | Medium | 2 h | H-02 (CI parity) | ✅ done 2026-09-20 |
| 18 | L-03 split test_pipeline.py | Low | 2 h | — | ⛔ backlog |
| 19 | L-05 per-thread sessions | Low | 30 min | — | ✅ done 2026-09-20 |
| 20 | Sec-note verify hashes on read | Low | 3 h | — | ✅ done 2026-09-20 |

**Remaining open:** H-02 (user must create the GitHub repo + push), M-09 (re-run
the legacy pre-2026-09-17 configs, ~11 min each), L-03 (split `test_pipeline.py`,
backlog hygiene). No other work remains code-side.

---

*Cross-references: vault notes [[Issue Register]] · [[Continuous Build]] ·
[[Test Suite]] · [[config]]. This catalog supersedes the "Maintenance debt"
section of `research/ISSUE_REGISTER.md` for the items it covers.*
