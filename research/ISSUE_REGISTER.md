# Quobo — Issue Register (Post-Remediation, v2)

**Date:** 2026-09-10 · **State:** after executing AUDIT_REPORT.md (tag `v1.0-results`) + the open-issue queue (OPEN-1..10)
**Method:** every item verified against current code/artifacts. Resolved items listed with verification evidence; externally-blocked items listed with their exact blocker.

---

## Part 1 — Resolved (original audit + issue register, with verification)

| ID | Issue | Sev | Fix commit | Verification |
|---|---|---|---|---|
| VCS-001 | No git repo | Critical | 5d889c9 | 17 commits, `v1.0-results` tag, clean tree |
| QML-001 | MinMaxScaler fit on test | High | 5d889c9 | 2 dedicated tests green |
| PSI-001 | PSI ignored hazard weights | High | 5d889c9 | Tests green; bands spread 162→500 |
| MET-001 | No significance tests, n=5 | High | 5d889c9 | 78 Holm-corrected comparisons @ n=25 |
| LKG-002 | Same-photo crops in train+test | High | 011b099 | GroupShuffleSplit wired |
| OPEN-1 | numpy spec vs pinned env contradiction | High | d21fb62 | `numpy>=2.3,<2.6` in both specs == 2.5.2 actual |
| DOC-001 | README crash + phantom modules | Med | 5d889c9 | All documented commands re-run OK |
| CCH-001 | Unconditional stale feature cache | Med | 5d889c9 | Fingerprint auto-invalidation live-tested |
| DRY-001/SEED-001 | Duplicated retraining, fixed seed | Med | 011b099 | Confusion/dashboard consume predictions |
| PERF-001 | SA 5× wasted reads | Med | 5d889c9 | num_reads=repeats |
| OPEN-2 | Z_full fake timing; α unrecorded | Med | d21fb62 | α + trace + nudged_to_k in rep JSONs; Z_full `no_selection:true` |
| OPEN-3 | Unbounded image response | Med | d21fb62 | 20MB cap + 404/410 fail-fast |
| OPEN-4 | Dead config keys | Med | d21fb62 | Keys removed; load_config warns on unknowns |
| OPEN-5 | Transductive PCA | Med | 6c099e1 | **Leak-free protocol shipped**: per-split scaler+PCA from cached raw embeddings; full 25-repeat rerun done (10.7 min); paper updated to v2 numbers |
| OPEN-6 | TACO provenance unrecorded | Med | 9b91042 | sha256 ac1ec605…, n=1500 pinned in annotations_provenance.json |
| BBOX-001 | Clamp wrong space | Low | 5d889c9 | Analytic test (48×96) passes |
| PKG-001 | No packaging/unpinned deps | Med | 011b099+cf8d444 | `pip install -e .` + constraints.txt (150 pins) |
| SEC-001/2/3 | Download/path/HTML | Low | 5d889c9 | Caps + containment + escaping |
| TST-001 | Tautological tests | High(debt) | c656ab6 | **16 tests** (unit+integration+folder-mode), green |
| OPEN-8 | Folder-mode untested | Low | f8a6a37 | **Real bug found & fixed** (zone = grandparent dir); zone_dir column honored; 2 new tests |
| MI-BIN | MI bias caveat | Low | — | Documented in paper §5 |

## Part 2 — Remaining (externally blocked; nothing left code-side)

| ID | Item | Blocker | Ready-when |
|---|---|---|---|
| OPEN-7 | CI unexercised | No GitHub remote — **you** must create repo + push | `ci.yml` is complete; first push runs it |
| OPEN-9 | Real Gwalior map | Ward-boundary GeoJSON — **you** must obtain (GMC/Smart City SPV/GADM) | Analytics ready; swap CSS grid for Folium when GeoJSON lands |
| OPEN-10 | Cross-dataset replication | TrashNet ~2GB zip download (GitHub API rate-limited this session) + extraction | **Harness complete & proven**: `crops_dir_override` config key + `configs/experiment_trashnet.yaml` — one command after extraction |
| S3 | D-Wave QPU run | DWAVE_API_TOKEN — **you** must register at dwave-system.com/leap | `scripts/qpu_run.py` verified to build the QUBO and skip green without token |

## Part 3 — Final state

- **Tests:** 16/16 green · **Lint:** ruff clean · **Reproducibility:** verified (same seed → same selection)
- **Definitive numbers (leak-free v2, 25 group-aware repeats, Holm-corrected):** Full-50 48.3% > LASSO-8 41.4% ≈ mRMR-8 41.4% ≈ **QUBO-8 40.9%** ≈ PCA-8 40.1% ≈ MI-8 39.5% (all principled arms tie, p>0.5) ≫ Random-8 32.2% (p<0.001). QSVC ~37% (kernel parity). All principled selection methods match under the strict protocol — QUBO additionally ships a globally-optimal formulation that runs unchanged on annealers.
- **Paper:** research/PAPER_DRAFT.md on v2 numbers, with v1-vs-v2 protocol materiality reported transparently.
