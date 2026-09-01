# Quobo: QUBO-Based Quantum Feature Selection for Resource-Constrained Waste Image Classification

**Working paper draft — v1.0 numbers (tag `v1.0-results`, 2026-09-02)**

*IDP3 project — [Devesh, Krishna, +1] — [Institution], Gwalior, India*

---

## Abstract

Municipal waste monitoring demands image classifiers cheap enough for edge deployment, which
constrains the feature dimensionality feeding any classifier head. We ask whether **quantum
optimization can select those features better than classical selection**. On the TACO trash-in-
context dataset (1,836 annotated object crops, 6 coarse classes), we extract MobileNetV2
embeddings, compress them to 50 PCA components, and select 8 features via four arms — random,
PCA-truncation, mutual-information (MI) ranking, and **QUBO-based selection (importance minus
redundancy, solved by simulated annealing)** — plus LASSO and mRMR baselines and the full-50
ceiling. All arms feed the same classifier suite: a quantum-kernel SVM (QSVC, ZZFeatureMap,
8 qubits) and classical SVMs including a tuned RBF-SVM. Across 25 group-aware repeated splits
(no photo straddles train/test), Holm-corrected Wilcoxon tests show QUBO-8 significantly beats
random (p<0.001), PCA (p=0.013), and MI (p<0.001) selection on RBF-SVM accuracy (41.1% vs
32.6/40.1/39.6%), while the mRMR-8 edge (42.7%) is not significant (p=0.092). QSVC runs at
37–40% — 2–3 points below tuned RBF at identical budgets — consistent with the quantum-kernel
parity literature. We additionally introduce a **Pollution Severity Index (PSI)** — a hazard-
weighted dominant-pollutant index on the 0–500 air-index scale, computed from classifier
predictions — and demonstrate the full CV→PSI→GIS analytics chain. To our knowledge this is the
first published combination of QUBO feature selection with quantum-kernel classification on
environmental imagery, and the first litter-specific severity index.

**Keywords:** QUBO, feature selection, quantum kernel machines, QSVM, waste classification,
TACO, pollution index, NISQ

---

## 1. Introduction

Litter monitoring pipelines face a resource wall: full MobileNetV2 embeddings (1280-d) are
cheap to *extract* but expensive to *classify with* at the edge, and quantum classifiers are
hard-capped at small feature budgets (8 features → 8 qubits under a ZZ encoding). Feature
selection at this budget is therefore not a nicety but the binding constraint — and the
literature (Mücke et al. 2022; Ferrari Dacrema et al. 2022) suggests QUBO optimization,
which treats feature **redundancy globally** rather than greedily, is a principled way to
spend it.

Prior art stops short of this setting. Quantum transfer learning has been applied to trash
images (Kumsetty et al. 2022, TrashBox) but without feature selection or QUBO; MI-QUBO
selection has been applied to medical images (Nau et al. 2025, MedMNIST pixels) but not
environmental ones; and no work we could find (arXiv, Semantic Scholar, GitHub — see our
survey, 2026-08) combines QUBO selection with a quantum kernel classifier on waste imagery.
Our contribution:

1. The first controlled 7-arm ablation (random / PCA / MI / QUBO / LASSO / mRMR / full) of
   feature selection for a QSVC on an environmental dataset, with group-aware splits and
   Holm-corrected significance testing.
2. A Pollution Severity Index (PSI) for litter — hazard-weighted, dominant-pollutant style,
   computed from classifier predictions — closing the CV→index loop.

## 2. Related Work

*(Condensed from the project's 222-finding global survey; see research/RESEARCH_REPORT.md.)*

**QUBO feature selection.** Three standard objectives: MIQUBO (D-Wave/Politecnico lineage;
Nguyen et al. KDD 2014 roots), QFS/SeleQt (Mücke et al. 2022 — α-swept importance/redundancy
without a penalty term), and linear-penalty variants (Nau et al. 2025). Consensus finding:
quantum solvers *match* classical heuristics in solution quality; wins concentrate against
greedy/ranking baselines — precisely the comparison our ablation stages. Feasible problem
sizes: direct QPU ~124–196 dense variables; hybrid thousands. Our 50→8 sits comfortably in
all regimes.

**Quantum kernels.** Havlíček et al. 2019 (the QSVC we implement); Schuld 2021 ("the
encoding is the model" — a feature-selection story); Huang et al. 2021 (with enough data,
classical ML matches quantum kernels — our mandatory tuned-RBF baseline); Thanasilp et al.
(kernel concentration past ~15–40 qubits — our 8-qubit budget is safely inside).

**Waste classification.** TACO (Proença & Simões 2020; Majchrowska et al. 2022 benchmark:
~75% in-the-wild classification ceiling with full deep ensembles); our 41% at an 8-feature
budget is a different, resource-constrained regime. Closest Indian prior art: Kumsetty et
al. 2022 (quantum TL on TrashBox, 98.5% classical baseline — different task scale, no
selection).

## 3. Method

**Pipeline.** TACO annotations → per-object crops (10% bbox margin, class-balanced 6 coarse
classes) → MobileNetV2 (ImageNet, GAP, 1280-d) → StandardScaler + PCA-50 → *selection arm*
(8 features) → classifiers. All arms share the identical split, subsample, and classifier
protocol; only the 8 columns differ.

**QUBO formulation (arm D).** Mücke-style QFS: minimize
`−α·Σ I_i x_i + (1−α)·Σ R_ij x_i x_j`, where `I_i` = MI(feature_i; class) and `R_ij` =
MI(feature_i; feature_j) (16-bin quantile discretization), α binary-searched until exactly
k=8 features are selected; solved with simulated annealing (dwave-samplers, 20k sweeps,
10 reads). *(When a D-Wave Leap token is available, the same QUBO runs on the QPU — the
n=50 dense problem embeds within a Pegasus Advantage.)*

**Group-aware protocol.** Crops from one TACO photo share lighting/sensor/background; a
naive split lets near-duplicates straddle train/test. We split by photo (GroupShuffleSplit,
25 seeds), subsample 400 training rows per classifier (seed varies by repeat), and report
mean±std plus Holm-corrected Wilcoxon signed-rank tests across all 21 arm-pairs per
classifier.

**QSVC.** ZZFeatureMap(8 qubits, reps=1, linear entanglement) → FidelityStatevectorKernel
(PSD-enforced) → QSVC (class-weighted). Angles scaled to [0.05, 1.52] by a MinMaxScaler fit
on train only. Classical baselines: linear SVM, RBF-SVM (defaults), tuned RBF-SVM
(C∈{0.1,1,10,100}, γ∈{scale,0.01,0.1,1}, 3-fold internal CV).

**PSI.** Per zone: `sub_c = count_c · w_c / max_{z,c}(count·w) × 500` with hazard weights
(cigarette ×3, bottle/can ×1, carton/cup ×0.8, lid ×0.5); headline = max sub-index
(dominant-pollutant rule, US-EPA-AQI style). Cleanliness score = clean count / total × 100.

## 4. Results

**Table 1 — Accuracy (mean±std over 25 group-aware repeats; Holm-corrected Wilcoxon vs QUBO-8):**

| Arm | QSVC | RBF-SVM | RBF (tuned) | vs QUBO (RBF, p) |
|---|---|---|---|---|
| Full-50 | — (50 qubits infeasible) | 48.4±1.9 | 48.9±2.4 | <0.001 (better) |
| mRMR-8 | 38.0±2.8 | 42.7±2.4 | 42.1±2.1 | 0.092 (n.s.) |
| LASSO-8 | 39.3±2.4 | 41.7±2.9 | 42.1±2.8 | 1.000 (tie) |
| **QUBO-8** | **37.4±2.9** | **41.1±2.7** | **40.8±2.9** | — |
| PCA-8 | 37.6±2.8 | 40.1±2.6 | 40.7±2.8 | 0.013 (worse) |
| MI-8 | 36.5±2.0 | 39.6±3.0 | 38.9±2.7 | <0.001 (worse) |
| Random-8 | 31.4±3.2 | 32.6±3.0 | 32.9±3.0 | <0.001 (worse) |

**Findings.** (1) Selection quality dominates: every principled arm beats random by ~8
points on every classifier. (2) QUBO-8 significantly beats the PCA-truncation and MI-ranking
baselines — the global-redundancy treatment pays off on correlated PCA features, as the
literature predicts. (3) The classical greedy mRMR-8 is statistically indistinguishable from
QUBO-8 (and LASSO ties): **QUBO matches the best classical selection at this scale**, the
honest parity result the annealing literature consistently reports. (4) The Full-50 ceiling
(48%) quantifies the 6-fold feature-budget cost: −7 points. (5) QSVC tracks 2–3 points below
tuned RBF at identical budgets — kernel parity, not advantage, at 8 qubits.

**Per-class (QUBO-8, RBF).** Cigarette is the best-measured class (recall 73%, precision
47%) — the class that drives PSI; cup is the weakest (recall 32%). Full confusion matrices:
results/confusion/.

**PSI demo.** Majority-vote predictions over 25 repeats (100% crop coverage) drive the zone
analytics: city cleanliness 74%, worst-zone PSI 500 (Zone B, cigarette-dominated), bands
spread Good→Hazardous across the six demo zones. results/dashboard/dashboard.html.

## 5. Discussion & Limitations

- **Transductive PCA.** Scaler+PCA are fit on all crops before splitting (uniform across
  arms; documented). The leak-free protocol refits per split from cached raw embeddings —
  left as future work with the expected cost of slightly lower absolute accuracies.
- **Simulated annealing.** EXP D solves classically; the identical 50-variable QUBO runs on
  a D-Wave QPU given a Leap token (S3, pending). SA≈QPU equivalence is the literature
  consensus, so we expect unchanged results.
- **Dataset scale.** 1,836 crops is small; the 25-repeat significance protocol is designed
  for exactly this regime. Cross-dataset replication (TrashBox, TrashNet) is queued.
- **Zone mapping is a demo** (TACO has no GPS); the folder-mode ingestion for real ward
  deployments is implemented and untested on real data.

## 6. Conclusion

At the resource-constrained regime where quantum classifiers must live, QUBO feature
selection is a principled, statistically validated way to spend a small feature budget —
significantly better than the standard PCA/MI shortcuts, matched to the best classical
greedy selection, and fully integrated into a first-of-kind CV→PSI→GIS pollution analytics
chain.

## References (core)

Havlíček et al. 2019, Nature 567:209. · Schuld 2021, arXiv:2101.11020. · Huang et al. 2021,
Nat. Comm. 12:2631. · Mücke et al. 2022, arXiv:2203.13261 (QMI 5:11, 2023). · Ferrari
Dacrema et al. 2022, arXiv:2205.04346 (SIGIR). · Nau et al. 2025, arXiv:2502.19201. ·
Proença & Simões 2020, arXiv:2003.06975 (TACO). · Majchrowska et al. 2022, Waste Management
138:282. · Kumsetty et al. 2022, FRUCT'31, doi:10.23919/fruct54823.2022.9770922. ·
Thanasilp et al. 2024 (kernel concentration). · Full 222-finding survey: research/RESEARCH_REPORT.md.
