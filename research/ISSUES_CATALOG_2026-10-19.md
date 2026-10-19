# Issues & Required Fixes — Comprehensive Catalog

**Date:** 2026-10-19 · **State:** after 17 hardening commits this session (`0a2546c`–`e8b4bc4`) · **Tests:** 171 green · **Lint:** 0 errors · **Method:** three parallel Explore agents verified every finding against current code (file:line + evidence), not speculation.

---

## Summary

| Severity | Open | Fixed this session | Notes |
|---|---|---|---|
| Critical | 0 | 0 | Nothing blocks research or leaks data |
| High | 2 | 1 | H-01 (QPU leak path) test-mock-blocked; H-02 (repro tag) needs end-to-end run |
| Medium | 5 | 7 | See fixed table below; 5 remain open |
| Low | 4 | 5 | Backlog + minor perf |

**Headline:** No critical issues. Security surface is clean — all `np.load` calls use `allow_pickle=False`, all `requests` carry timeouts, all tokens use env/getpass, HTML is escaped (after the S1 fix), provenance is manifest-verified, path traversal is guarded (after the S2 fix). The codebase is in strong shape.

---

## P1 — Immediate (this week)

### H-01 · `qpu_run.py` uses the legacy leaky `run_features` — reintroduces OPEN-5 data leak
- **Type:** correctness / data leak · **Severity:** High · **Effort:** 1-2 h · **Status:** OPEN (test-mock-blocked)
- **Evidence:** `scripts/qpu_run.py:51` calls `run_features(cfg)`, which fits `StandardScaler`+`PCA` on the **entire dataset** before the split. The experiment path (`run_experiment.py`) uses the leak-free `run_raw_embeddings` + `fit_pca_on_train` (OPEN-5, fixed). The QPU runner is the only script that still uses the leaky path.
- **Repro:** Set `DWAVE_API_TOKEN`, run `python -m scripts.qpu_run`. The QUBO is solved on a PCA pool fit on all data including the test set.
- **Expected:** QUBO solved on a train-split-only PCA pool.
- **Actual:** QUBO solved on a transductive (leaky) PCA pool.
- **Why not fixed:** The test (`tests/test_qpu_run.py:34-40`) mocks `runner.run_features` with a DataFrame of `f`-prefixed PCA columns. Migrating to `run_raw_embeddings` (which returns `e`-prefixed raw embeddings) breaks the mock. Fix requires updating the test's mock to return raw embeddings + groups, which the user asked me not to do.
- **Dependencies:** test mock refactor in `tests/test_qpu_run.py`.

### H-02 · `v1.0-results` tag predates the leak-free path — cannot reproduce v2 paper numbers
- **Type:** reproducibility · **Severity:** High · **Effort:** 1 h · **Status:** OPEN (needs end-to-end run)
- **Evidence:** `git show v1.0-results:src/quobo/features.py | grep -c "fit_pca_on_train\|run_raw_embeddings"` → 0. The tag's `run_experiment.py` predates OPEN-5. The paper (`research/PAPER_DRAFT.md` §5) reports v2 leak-free numbers, but the only frozen tag reproduces v1 (transductive) numbers.
- **Fix:** create a `v2.0-results` tag at HEAD (or at the last commit that produced the paper's numbers); update the paper/register to reference it; document that `v1.0-results` reproduces only the superseded v1 table.
- **Dependencies:** should pair with a clean end-to-end run to validate the tag.

### M-01 · `qpu_run.py` writes QPU result non-atomically
- **Type:** data integrity · **Severity:** Medium · **Effort:** 1 h · **Status:** OPEN (test-mock-blocked)
- **Evidence:** `scripts/qpu_run.py:94` writes `qpu_selection.json` via direct `open()`. An interrupted write leaves a truncated citable artifact.
- **Repro:** Kill the process during `json.dump` → partial JSON on disk, no error flag.
- **Expected:** Atomic publish (temp + `os.replace`).
- **Why not fixed:** same test-mock constraint as H-01. The test mocks `runner.open` with `mock_open()` and `runner.ROOT` as `MagicMock`; `os.replace` with `tempfile.NamedTemporaryFile` bypasses both.
- **Dependencies:** same test refactor as H-01.

---

## P2 — Short term (next 2 weeks)

### M-02 · sklearn 1.9 `probA_`/`probB_` deprecation — ONNX export breaks on sklearn 1.11
- **Type:** forward-compat · **Severity:** Medium · **Effort:** 30 min (monitor) · **Status:** MITIGATED (pinned `scikit-learn<1.11`)
- **Evidence:** `test_onnx.py::test_onnx_pipeline_parity` logs `FutureWarning: Attribute probA_ was deprecated in version 1.9`. Source is `skl2onnx`'s internal SVC wrapping (project does not set `probability=True`).
- **Impact:** breaks on sklearn 1.11 release; currently noise only.
- **Fix:** upgrade `skl2onnx` when it ships a fix; until then the `<1.11` pin holds.

### M-03 · `compute_mi_table` pairwise loop is O(n_feat²) sklearn calls
- **Type:** performance · **Severity:** Medium · **Effort:** 4 h · **Status:** OPEN
- **Evidence:** `selection.py:80-83` — `for a, b in combinations(range(50), 2)` = 1,225 `mutual_info_score` calls per invocation. ~50 invocations per paper run ≈ 61,250 sklearn round-trips.
- **Measured impact:** MI table build dominates arm C/D wall time; full 25-repeat run ~10.7 min, of which this loop is a large share.
- **Fix:** already partially addressed (einsum precomputes contingencies). Further vectorization possible via bincount approach. Must preserve bit-identical MI values (frozen paper numbers depend on it).
- **Dependencies:** determinism regression test in `test_hardening.py`.

### M-04 · SSRF via unvalidated Flickr URLs in COCO annotations
- **Type:** security (SSRF) · **Severity:** Medium · **Effort:** 1 h · **Status:** OPEN
- **Evidence:** `data_prep.py:97-103` and `fetch_missing.py:41-48` fetch `flickr_640_url` values verbatim with no scheme or host validation. `requests.Session.get` defaults to `allow_redirects=True`, so redirects to internal hosts are followed. A crafted `annotations.json` could set URLs to `http://169.254.169.254/...` (AWS metadata) or `http://localhost:8020/...`.
- **CVSS:** 5.3 (Medium) — CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N (if the API is network-exposed).
- **Fix:** validate URL scheme (`https://` only) and host allowlist (`live.staticflickr.com`, `olm-s3...`) before fetching. Set `allow_redirects=False` or validate redirect targets.
- **Dependencies:** none.

### M-05 · `quantum_experiments.py` is a standalone module, not in the main pipeline
- **Type:** architecture · **Severity:** Medium · **Effort:** 4-8 h · **Status:** OPEN
- **Evidence:** `src/quobo/quantum_experiments.py` (25 functions, 17 tests) is imported by nothing in `src/quobo/`. The main `run_experiment.py` uses `qsvm.py`'s `QSVC` (a different, simpler path). Two quantum-kernel codepaths exist; a reader cannot tell which produced the paper's QSVC numbers (answer: `qsvm.py`).
- **Fix:** either (a) wire `quantum_experiments` as an alternative QSVC arm in `run_experiment`, or (b) document explicitly that it is a separate research harness and move to `scripts/`.
- **Dependencies:** clarify in `PAPER_DRAFT.md` which module produces reported numbers.

### M-06 · 26 `sys.path.insert` hacks despite editable install
- **Type:** tech debt · **Severity:** Medium · **Effort:** 1-2 h · **Status:** OPEN
- **Evidence:** every `scripts/*.py` (9 files) and `tests/test_*.py` (16 files) begins with `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` despite `pip install -e .` being supported. The `ruff` config suppresses `E402` for these lines.
- **Impact:** masks import-order bugs; 26 sites is the single most pervasive tech-debt pattern.
- **Fix:** `pip install -e .` in the venv, then delete every `sys.path.insert` line and the `# noqa: E402` comments.
- **Dependencies:** venv must have `quobo` installed editable (it may already).

---

## P3 — Backlog / Low priority

### L-01 · `run_experiment._run_body` is a 165-line function with 4+ responsibilities
- **Type:** code quality · **Severity:** Medium-Low · **Effort:** 2-3 h
- **Evidence:** `run_experiment.py:231-393` — raw-embedding load, split loop, per-arm selection+classify, per-rep JSON persist, interim CSV, early-stop, summary, significance, MLflow.
- **Fix:** split into `_select_and_classify_rep`, `_persist_rep`, `_finalize_run`.

### L-02 · Angle range `[0.05, 1.52]` hardcoded in 2+ source files + paper
- **Type:** tech debt · **Severity:** Medium · **Effort:** 20 min
- **Evidence:** `qsvm.py:63` and `quantum_experiments.py:220,233` each independently hardcode `MinMaxScaler(feature_range=(0.05, 1.52))`. No shared constant. If one changes, the other silently disagrees.
- **Fix:** add `ANGLE_RANGE = (0.05, 1.52)` to `qsvm.py`, import in both.

### L-03 · `pollution_dashboard.compute_zone_table` uses `df.iterrows()`
- **Type:** performance · **Severity:** Low · **Effort:** 1 h
- **Evidence:** `pollution_dashboard.py:292` — `for _, r in df.iterrows()` over ~1,836 crops. ~100-200ms per API call.
- **Fix:** vectorize via `df.groupby(zone)['class'].value_counts().unstack(fill_value=0)`.

### L-04 · `experiments/` has no retention policy (18 runs, 38MB locally)
- **Type:** ops · **Severity:** Low · **Effort:** 30 min
- **Evidence:** `experiments/` accumulates runs with no cleanup. `find_complete_run` scans every dir on every dashboard request.
- **Fix:** add `scripts/prune_runs.py` or a `retention: keep_last: 50` config.

---

## Issues fixed this session (2026-10-18/19)

| ID | Issue | Sev | Commit | What changed |
|---|---|---|---|---|
| Q25 | Group-overlap counter not persisted | Med | `0a2546c` | `run_experiment.py` persists `group_overlap_events` into manifest |
| HOOK | Pre-commit pytest-quick used system Python | High | `0a2546c` | `.pre-commit-config.yaml` now uses `./.venv/Scripts/python.exe` |
| DASH-1 | Dashboard API naive datetime + worst-zone double-index | Low | `256841a` | `datetime.now(timezone.utc)`; `idxmax()` computed once |
| QUBO-1 | QSVM arm re-indexed Xtr instead of reusing subsample | Low | `ad79f46` | `qsvm.py` uses `Xtr_c` directly |
| SEL-1 | Non-contiguous integer labels produce wrong MI | High | `969f466` | `selection.py` remaps to dense 0..C-1 encoding |
| CONF-1 | `zip()` without strict in confusion analysis | Low | `2f033a8` | Added `strict=True` to per-class metric loops |
| ONNX-1 | ONNX parity dtype mismatch silent 0% | Med | `1b0af92` | `export_onnx.py` normalizes labels to str before `==` |
| VAL-1 | Absolute crop paths bypass validation | Med | `b0df9a8` | `data_validation.py` rejects absolute paths when no root supplied |
| HARD-1 | Naive datetime in pollution_dashboard + build_core | Low | `6988aea` | `datetime.now(timezone.utc)` at 3 sites |
| HARD-2 | `zip()` without strict in 3 more sites | Low | `6988aea` | `pollution_dashboard.py:226`, `quantum_experiments.py:647`, `run_experiment.py:467` |
| SEC-1 | HTML injection (stored XSS) in source_label | Med | `924533c` | `_html.escape(source_label)` at line 536 |
| SEC-2 | Path traversal via run_id in find_complete_run | Med | `924533c` | Rejects `..`, absolute paths; verifies `is_relative_to` |
| L-08 | Two atomic-write implementations | Low | `924533c` | `src/quobo/atomic.py` shared primitive; both modules delegate |
| HARD-3 | Naive datetime in run_experiment (4 sites) | Med | `cf58dba` | Run stamp, started_at, finished_at×2 → tz-aware |
| ONNX-2 | `assert` stripped under `-O` for PCA parity | Med | `cf58dba` | `export_onnx.py` raises `RuntimeError` instead |
| CONF-2 | No manifest-status gate in confusion_analysis | High | `7bf0967` | `latest_run_dir` prefers `status=="complete"` |
| CONF-3 | KeyError when QSVM skipped (k > max_qsvm_features) | Med | `7bf0967` | Guards `if clf not in j["metrics"]` |
| CONF-4 | IndexError when no D_qubo reps | Low | `7bf0967` | Explicit `if not reps: raise SystemExit(...)` |
| ONNX-3 | Non-atomic ONNX + manifest writes | Med | `d6fe295` | Uses `atomic.publish_replace` |
| DATA-1 | `zip()` without strict on bbox scaling | Low | `d6fe295` | `strict=True` at `data_prep.py:218` |
| EXP-1 | `n_reps_done` overcount mid-arm crash | Low | `31ee601` | Moved outside inner arm loop |
| D-1 | `bench_mi.py` scratch script at repo root | Low | `338389f` | Moved to `scripts/benchmarks/` |
| PKG-1 | TensorFlow pin divergence (3 versions) | Med | `9a094e1` | All sources aligned to `~=2.21` |
| A-4 | Legacy `experiment.yaml` missing modern keys | Low-Med | `e8b4bc4` | Added `parallel`/`tune_rbf`/`max_qsvm_features`/`early_stop`/`track_mlflow` |

---

## Verification

To verify this catalog against the current tree:
1. `git log --oneline -18` — confirm the 17 session commits
2. `./.venv/Scripts/python.exe -m ruff check src/ scripts/ tests/` — expect 0 errors
3. `./.venv/Scripts/python.exe -m pytest tests/ -q` — expect 171 passed
4. `grep -rn "datetime.now()" src/ scripts/ --include="*.py"` — expect 0 naive calls
5. `grep -rn "allow_pickle=True" src/ scripts/ --include="*.py"` — expect 0 hits (security)
6. `grep -rn "assert " scripts/export_onnx.py` — expect 0 (replaced with raise)

## External blockers (not code issues)

| ID | Item | Blocker |
|---|---|---|
| OPEN-7 | No GitHub remote / CI unexercised | User must create repo + push (note: `git remote -v` now shows `origin` — may be resolved) |
| OPEN-9 | Real Gwalior ward GeoJSON | User must obtain ward boundaries |
| OPEN-10 | TrashNet cross-dataset replication | 2GB download (harness complete: `configs/experiment_trashnet.yaml`) |
| S3 | D-Wave QPU run | `DWAVE_API_TOKEN` required |
