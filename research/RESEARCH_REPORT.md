# Global Research Report
## QUBO-Based Quantum Feature Selection + QSVM for Environmental (Waste) Image Classification

**Project:** IDP3 — Quobo | **Team:** Devesh, Krishna + 1 | **Location:** Gwalior, India
**Date:** 2026-08-25 | **Method:** 8 parallel research agents + adversarial gap-check critic; ~1,000 web/fetch operations; 222+ verified findings; every load-bearing number cross-checked against primary sources (arXiv full texts, GitHub repos, official dataset pages).

---

## 1. Executive Summary

**The headline: your exact pipeline has never been published.** Exhaustive searches across arXiv, Semantic Scholar, Crossref, GitHub, and the open web found **zero papers combining QUBO-based feature selection with a quantum SVM on waste/trash imagery**. Two Indian groups did quantum *transfer learning* on trash images in 2022 (NIT Karnataka's TrashBox paper; an AIP Nagpur conference paper) — those are your closest prior art and must be cited and differentiated. Your specific contribution window — **TACO → MobileNetV2 → PCA → QUBO-selects-8 → QSVM, with a random/PCA/MI ablation** — is empty.

**Second headline: your "Pollution Severity Index" is genuinely novel.** The term does not exist for litter/waste anywhere in indexed literature (existing "PSI" uses are Singapore's Pollutant *Standards* Index and heavy-metal geoaccumulation indices). And zero quantum + GIS/dashboard work surfaced anywhere. Even your analytics layer is first-of-kind.

**Third headline: the honest scientific expectation is parity, not dominance.** Every rigorous study concludes quantum solvers currently *match* classical heuristics on QUBO feature selection, and QSVM typically matches or slightly beats tuned RBF-SVM (±1 point at 5-8 features). The defensible claims are: global treatment of feature redundancy (escaping greedy local minima), solution quality at small feature budgets, parameter efficiency, and a scalability runway. Frame it that way and reviewers will not object.

**Feasibility: your design sits exactly in the demonstrated sweet spot.** 8 features → 8 qubits is proven on real hardware; ~50 candidate features is comfortably within both annealer (direct QPU caps ~124-196 dense variables; hybrid handles thousands) and simulator reach; kernel concentration (the main theoretical failure mode) only becomes dangerous past ~15-40 qubits.

---

## 2. Novelty Assessment (the core claim, verified three ways)

| Search route | Result |
|---|---|
| arXiv API: quantum + waste/trash/garbage/litter classification | **0 hits** (re-checked independently by the critic) |
| Semantic Scholar / Crossref: quantum + waste imagery | **2 hits**, both quantum *transfer learning*, both Indian, both 2022 |
| GitHub: TACO/TrashNet + QSVM/QUBO | **0 repos** (one marginal 0-star QCNN-on-TrashNet notebook) |

**Closest prior art (cite and differentiate):**
1. **Kumsetty et al. 2022**, "TrashBox: Trash Detection and Classification using Quantum Transfer Learning," IEEE FRUCT'31 (NIT Karnataka, India) — [doi:10.23919/fruct54823.2022.9770922](https://doi.org/10.23919/fruct54823.2022.9770922), ~45 citations. CNN features + variational quantum circuit on TrashBox; classical TL baseline 98.47%. **No feature selection, no QSVM, no QUBO, not TACO.**
2. **Mogalapalli et al. 2022**, "Trash classification using quantum transfer learning," AIP Conf. Proc. 2424 (ICCICA-21, Nagpur, India) — [doi:10.1063/5.0076837](https://doi.org/10.1063/5.0076837). Same pattern, paywalled.
3. **Nau et al. 2025**, "Quantum Annealing Feature Selection on Light-weight Medical Image Datasets" — [arXiv:2502.19201](https://arxiv.org/abs/2502.19201) (FAU Erlangen, Germany + UCL, UK). **The direct methodological analogue**: MI-QUBO selecting 25 of 784 pixels on MedMNIST. Right method, wrong domain (medical, not environmental).
4. **Felefly et al. 2023** — QUBO(D-Wave)-selected radiomic features → 2-qubit QNN on brain MRI. Same architecture pattern, wrong domain.

Your novelty sentence for the paper: *"No published work combines QUBO-based feature selection with a quantum SVM on environmental imagery; we present the first, with a controlled four-arm ablation (random / PCA / MI / QUBO at identical 8-feature budget)."*

---

## 3. Method Foundations: QUBO Feature Selection

### 3.1 The three standard formulations (pick one, cite all)

1. **MIQUBO** (D-Wave/1QBit/Politecnico di Milano lineage, roots in Nguyen et al. KDD 2014 and Neven's QBoost 2012): diagonal = −MI(feature; label), off-diagonal = redundancy (−CMI or −|corr|), plus cardinality penalty **λ·(Σxᵢ − k)²**. Reference implementation: SIGIR 2022 paper + [qcpolimi repos](https://arxiv.org/abs/2205.04346).
2. **QFS / SeleQt** (Mücke, Heese, Müller, Wolter, Piatkowski — TU Dortmund + Fraunhofer, [arXiv:2203.13261](https://arxiv.org/abs/2203.13261)): minimize −α·ΣIᵢxᵢ + (1−α)·ΣRᵢⱼxᵢxⱼ where a single scalar α ∈ [0,1] *provably* controls subset size — binary-search α until exactly k=8 features are selected. **Cleanest for your project: no penalty term, no soft-constraint leakage.** Installable as `pip install seleqt`.
3. **Linear-penalty variant** (Nau et al. 2025): diagonal-only |Σx−k| penalty preserves Q-matrix sparsity — the pattern to use if you ever embed on a real QPU.

### 3.2 Scale reality check (where your 50→8 sits)

| Solver | Max practical candidate features | Source |
|---|---|---|
| Direct D-Wave QPU (dense QUBO) | ~124 (196 with subsampling tricks) | SIGIR22; Nau 2025 |
| D-Wave Leap Hybrid | 3,000–8,000 | Milano CQFS |
| QAOA on IBM (simulator/hardware) | ≤ 21–32 | Turati QCE 2022; Hellstern |
| Simulated annealing (dwave-samplers) | thousands | universal baseline |

At ~50 candidates you are comfortably inside *every* regime — you can even brute-force-verify the QUBO optimum (exhaustive 2⁵⁰ is too much, but qbsolv/SA will find and certify near-optima; `seleqt` brute-forces to ~30 dims).

### 3.3 Expected result pattern for your ablation (from published precedents)

- **QUBO-8 vs MI-top-8**: QUBO should edge out MI ranking *when candidate features are mutually correlated* — and PCA/MobileNetV2 features are highly correlated. When features are independent, they tie. This is your expected positive result, and it's exactly what SIGIR22's linear-filter-vs-QUBO tables and Romero's LASSO/RFR/mRMR benchmarks predict.
- **QUBO vs LASSO**: Texas A&M's synthetic study (10,000 obs × 50 features) recovered the true 5 source features at **100% vs LASSO's 40%** — the strongest published head-to-head with the LASSO family.
- **Quantum solver vs classical solver on the same QUBO**: statistical ties (SIGIR22: no dominance across 10k experiments; SA ≡ QPU in CQFS). **Develop with simulated annealing; reserve D-Wave QPU/hybrid minutes for the final EXP D numbers** (free-tier minutes exhaust fast — Milano explicitly warns this).
- Soft constraints leak: realized count can deviate from k (within ~10% in half of SIGIR22 cases). With Mücke's α-sweep you get exactly 8 — verify this in your pipeline.

---

## 4. QSVM / Quantum Kernels

### 4.1 Your 8-feature budget is proven feasible

- 8 features map 1:1 to 8 qubits under **ZZFeatureMap** (one qubit per feature). Real-hardware demos exist at exactly this scale: 4 qubits on IonQ Harmony (Suzuki et al.), 8 qubits for satellite imagery (Rodriguez-Grasa et al.).
- On a noiseless statevector simulator (`FidelityStatevectorKernel`) there is no hard limit — the cost is the O(n²) kernel matrix over samples.
- **Kernel concentration** (Thanasilp et al.) is the real scientific risk: kernel values collapse exponentially when the encoding is too expressive/entangled/noisy, making QSVM trivial. Your 8-qubit budget is safely below the empirical cliff (~15-40+ qubits depending on encoding). **This is also a positive argument for your thesis**: small, carefully-chosen feature sets are exactly what quantum kernels need — multiple independent groups report adding features does *not* improve quantum-kernel performance.

### 4.2 The practical recipe (verified against current Qiskit 2.5.2 / QML 0.9.1)

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

feature_map = ZZFeatureMap(feature_dimension=8, reps=1, entanglement='linear')
kernel = FidelityQuantumKernel(feature_map=feature_map, enforce_psd=True)
qsvc = QSVC(quantum_kernel=kernel)
```

- `entanglement='linear'` costs 7 ZZ pairs per rep vs full's 28 — lower concentration risk.
- Cache the kernel matrix (or subsample) — with thousands of TACO samples the O(n²) matrix dominates runtime.
- All four ablation arms route through the *same* QSVC; you swap only the 8-column input. Clean attribution.

### 4.3 Expected QSVM-vs-SVM numbers (set expectations now)

| Study | Dataset (features/qubits) | QSVC | Classical SVC |
|---|---|---|---|
| Villalba-Ferreiro et al., IJCNN 2025 | MNIST-PCA 3v5 (5/5) | **92.64%** | 91.32% |
| same | Iris (4/4) | 95.89% | **95.98%** |
| Hyperspectral Indian Pines (binary) | (—/—) | **78.0%** | 72.0% |

Expect **parity to +1 point**; hyperspectral-style gains are possible. **Mandatory caveat citation: Huang et al. 2021 (Caltech/Google)** — with enough data, classical ML matches quantum kernels even on quantum-native problems. Include a strong classical RBF-SVM on the same 8 features, or reviewers will attribute gains incorrectly.

**Foundations to cite:** Havlíček et al. 2019 (Nature 567:209, IBM), Schuld & Killoran 2019, Schuld 2021 (kernel theory: *the data encoding is the model* — which dovetails exactly with a feature-selection story), Huang et al. 2021.

---

## 5. The TACO Dataset (verified numbers, corrected claims)

- **Citable v1.0: 1,500 images / 4,784 annotations**, COCO format, 60 litter categories in 28 supercategories (+ Unlabeled litter). CC BY 4.0, Zenodo DOI [10.5281/zenodo.3587843](https://doi.org/zenodo.3587843), code MIT. Paper: [arXiv:2003.06975](https://arxiv.org/abs/2003.06975) (Proença & Simões, Univ. Surrey UK/Portugal) — **arXiv-only, never peer-reviewed**; pair your citation with the peer-reviewed Majchrowska et al. 2022 (Waste Management, Gdańsk Tech/Intel Poland, ~305 citations).
- **The living dataset is bigger**: the unofficial weekly COCO dump (fetched 2026-08-25) holds **3,831 images / 8,334 annotations**, plus 2,404 queued unlabeled. ⚠️ **State which snapshot you used** — one of our own researchers' "9,500 images" claim was refuted; count your download.
- **Extreme class imbalance**: cigarette alone is 13.9% of instances; dozens of classes have <20 examples (min 1); max:min ratio ~667:1. → Use coarse classes, weighted sampling, and **macro-averaged** metrics.
- **Taxonomy decision required**: 28-way is far above what 8 features can support. Precedented middle grounds: the paper's own **TACO-10** (9 supercategories + Other), or detect-waste's 7-class scheme (WiMLDS Poland). Recommend: TACO-10 or 7-class for the main experiment, report per-class.
- **Published ceilings (frame expectations)**: original paper Mask R-CNN ~15.9-17.6 AP; best modern: EfficientDet-D2 **61.05 AP@0.5 single-class but only 18.78 AP 7-class**; Majchrowska ~70 AP detection / ~75% classification with deep ensembles. Failure modes: tiny objects (cigarettes <64px), visual ambiguity (bottle-vs-can). Your task is post-crop classification — a different, easier regime — but expect zone-level PSI robustness to come from *aggregating many classifications per zone*, not per-image accuracy.
- ⚠️ tacodataset.org was unreachable during our entire survey (connection resets). Verify license terms and current totals when you download; record per-image Flickr attributions if redistributing.

### Alternative datasets (worldwide, for robustness checks)

| Dataset | Origin | Size | License | Notes |
|---|---|---|---|---|
| **TrashNet** | Stanford, USA | 2,527 imgs / 6 classes | MIT | white-background; TL accuracies 87-98% (inflated regime) |
| **TrashBox** | NITK Surathkal, **India** | 17,785 imgs / 7 classes | **none stated** ⚠️ | incl. e-waste/medical; tops out 89.62% (ResNeXt-101) |
| RealWaste | Australia | 4,752 imgs / 9 classes | — | landfill |
| GD (Garbage Dataset) | — | 12,259 imgs / 10 classes | — | 2026 |
| UAVVaste | Gdańsk, Poland | 772 imgs / 3,718 anns | Apache-2.0 | aerial |
| MJU-Waste | China | 2,475 imgs | — | RGB-D |
| ZeroWaste | USA (CVPR'22) | — | CC BY 4.0 | industrial sorting (TrashCan variant restricts commercial use) |
| Drinking Waste | — | — | CC0 | — |
| GlobalWasteData (GWD) | DFKI, Germany | large | — | 2026 integrated archive |
| pLitterStreet / RoLID-11K / StreetView-Waste (WACV'26) | various | >13k geotagged / 11k dashcam | — | **street-level geolocated — templates for your Gwalior zone mapping** |

---

## 6. Classical Waste-ML Landscape (your baselines' context)

- **Your exact classical pipeline is citable standard practice**: Jin et al. 2023 (Waste Management) — attention-MobileNetV2, PCA-compressed FC layer, 90.7% at 600 ms on a Raspberry Pi 4B; Xu et al. 2020 — PCA-reduced MobileNetV2 features → SVM, 6 garbage types. This legitimizes branch B (PCA-8) as a serious baseline, not a straw man.
- **The QUBO thesis has direct classical support**: Nguyen et al. 2025 — deep features + SVM/LR reach up to 100% on TrashNet while feature selection removes **>95% of dimensions with no accuracy loss**. Most deep-feature dimensions are redundant; that's precisely the redundancy QUBO selection exploits globally.
- **Accuracy bands (report against the right one)**: white-background single-item datasets yield 87-98%; in-the-wild is much harder (TrashBox ≤89.6%, TACO detection ≤26 AP, Pomerania ~75%). Your TACO numbers belong in the *second* band.
- **Resource-constrained framing is a mature literature you can join**: WasteNet 97% on Jetson Nano (Trinity College Dublin); Li & Grammenos 95.98% @ 4.7W (Aberdeen); GECM-EfficientNet 1.23M params @ 146 ms inside a physical bin (China). Pitch the 8-feature QSVM head as latency/power-friendly.
- **Industry publishes different metrics**: Greyparrot (UK, 250+ analyzers / 65+ facilities / 20+ countries), AMP (USA), ZenRobotics (Finland, 500+ categories) — none publish cross-dataset accuracy.
- **Methodological caution**: small waste datasets produce inflated scores (train-val gaps of ~19 points documented; several 100%-claims; augmentation explains +4.2 points in one study). Your repeated-split design with identical budgets across arms A-D would be *more rigorous than most published practice* — say so.

---

## 7. Quantum for Sustainability — Worldwide

**Direct precedents (sparse but real):** QSVM on environmental *data* exists — water quality (South Africa, [arXiv:2411.18141](https://arxiv.org/abs/2411.18141)), air quality kernels on IBM hardware ([arXiv:2607.20377](https://arxiv.org/abs/2607.20377)), hyperspectral land cover ([arXiv:2605.17587](https://arxiv.org/abs/2605.17587)). **None on waste imagery.**

**Reference-point results:**
- MBZUAI (Abu Dhabi, UAE, M. Shafique group): QLIF-CAST weather forecasting — 15.4% lower MSE, 94% faster training, verified on 156-qubit IBM Marrakesh.
- Hyperspectral quantum kernels: 78.0% vs 72.0% classical (Indian Pines binary).
- **Honest negative results are mainstream**: cloud-microphysics stress tests (Germany/Europe) and Lean-4-proven limits on quantum power flow show classical baselines frequently winning. Your ablation design aligns with this skeptical literature's expectations.
- Universal recipe in environmental QML: **heavy classical compression before the circuit** — cloud detection uses PCA→4 qubits; solar-panel detection caps at 8 qubits; waste-VQC uses 4 qubits. Your plan sits exactly inside the demonstrated 2-8 qubit regime.
- Transferable QUBO encoding tricks: Uzbekistan irrigation QUBO ([arXiv:2607.13374](https://arxiv.org/abs/2607.13374), 584 variables, IBM Heron); SPLIT-Q grid islanding (qubit-bounded sequential subproblems).

**National initiatives:** India NQM (₹6,003.65 cr, 2023-31, four T-Hubs since Sept 2024) funds infrastructure (IISc CEQT, C-DAC QSim), not environmental QML; Japan Moonshot formally links quantum to environment; EU Flagship has no green slate (closest: EQUIP-G gravimetry); US DOE Quantum Genesis centers on fault tolerance.

**India-specific:** academic QML-for-environment is nearly empty (only a 2019 Thapar/Patiala evapotranspiration MPS classifier). Startups: BosonQ Psi (quantum-inspired CAE), QNu Labs (security), Qkrishi (finance). ⚠️ QpiAI's domain was listed for sale during the survey — verify before citing. **IIT Hyderabad's 2021 D-Wave l0-subset-selection paper is the natural Indian citation anchor.** Filling the domestic gap is part of your contribution.

---

## 8. Pollution Indices & GIS (your analytics layer)

- **"Pollution Severity Index" for litter does not exist** — the term is yours to define. Existing collisions: Singapore's Pollutant *Standards* Index (add a disambiguation footnote), heavy-metal Igeo indices.
- **All major air indices share one formula**: piecewise-linear concentration → sub-index, headline = **max** sub-index, 0-500 scale (US EPA AQI, Singapore PSI, India CPCB NAQI — NAQI uniquely adds NH₃ and Pb). A defensible waste-PSI: weighted mix of hazard-class densities per km², mirroring NAQI sub-index logic, or the simple cleanliness percentage mapped per ward.
- **Your cleanliness score mirrors how India operationalizes cleanliness**: Swachh Survekshan (QCI/MoHUA, scores /12,500) — but SS relies on surveys; position your dashboard as a **zone-level, image-derived complement**. Gwalior baseline: 10,995/12,500 in SS 2024 (⚠️ component sub-scores don't reconcile across sources — cite the total only).
- **Gwalior motivation is unusually strong**: WHO 2016 rated it the world's #2 most polluted city with **waste burning explicitly cited** as a principal PM source; ~144 µg/m³ annual PM2.5 (3rd-highest in India); Smart Cities Mission Phase-II; NCAP-monitored (20-30% PM reduction target); 3-4 active stations (MPPCB City Center, Deen Dayal Nagar, Maharaj Bada, Phool Bagh) for validation.
- **Remote-sensing method**: Biermann et al. 2020 (Plymouth Marine Lab UK + Univ. Aegean Greece), *Sci Rep* 10:5364 — Sentinel-2 Floating Debris Index + NDVI + Gaussian NB, 86% plastics accuracy (⚠️ correct citation; earlier drafts in our own survey had it wrong). ESA SNAP implements FDI as band arithmetic; QGIS zonal statistics/interpolation turns detections into per-zone surfaces.
- **Calibration data exists**: OpenLitterMap (Cork, Ireland; ODbL; 500k+ observations, 110+ countries), NOAA/UGA Debris Tracker (open). Litterati (US) is partnership-gated but its SF cigarette-tax case ($4M/yr) is the policy precedent.
- **Negative finding with strategic value**: NASA GLOBE Observer has no litter protocol; no space agency offers a citizen-trash product. Your CV→QUBO/QSVM→zone-index→GIS chain occupies an unfilled niche. No official QGIS/SNAP waste tutorial exists — your chain is self-assembled (novel, but re-derive index definitions from primary sources).

---

## 9. Global QML Research Landscape (by country)

| Country | Key players relevant to your project |
|---|---|
| USA | IBM Quantum, Google Quantum AI, Texas A&M CAILab, MIT, IonQ, Baylor, UVA, Missouri, Kentucky/ORNL |
| Canada | **D-Wave** (Ocean SDK, Leap, Advantage), Xanadu/PennyLane, Vector Institute |
| UK | STFC Hartree Centre (co-maintains Qiskit ML), Phasecraft, Univ. Surrey (TACO), Plymouth Marine Lab (FDI) |
| Italy | **Politecnico di Milano** (Ferrari Dacrema/Cremonesi/Ferro — the most prolific QUBO-FS group; all reference code), Padua, Torino, ESA Φ-lab |
| Germany | Fraunhofer IAO/IAIS/ITWM/IAF, TU Dortmund (SeleQt), FAU Erlangen (MedMNIST MI-QUBO), Forschungszentrum Jülich, DFKI (GlobalWasteData), Kipu Quantum |
| EU other | Jagiellonian + Silesian (Poland), Gdańsk Tech (TACO benchmarks), EHÜ/BCAM + Jaen (Spain — satellite quantum kernels), Univ. Oulu (Finland, ADAPT-QAOA), Trinity College Dublin (WasteNet) |
| UAE | MBZUAI (Shafique — environmental QML reference results) |
| China | Baidu IQC/Paddle Quantum, Zhejiang, USTC |
| Japan | Fujitsu, NTT, RIKEN; trapped-ion QSVC on MNIST (Suzuki); agri quantum kernels |
| Australia | Q-CTRL, Univ. Sydney, RMIT (Niu/Ren — most prolific 2024-26 in QUBO-FS), RealWaste |
| **India** | IIT Hyderabad (D-Wave l0-selection 2021), NIT Karnataka (TrashBox quantum TL), BosonQ Psi, QNu Labs, Qkrishi; IISc CEQT + C-DAC (NQM infra). **No environmental QML applications — your domestic gap.** |
| Pakistan | FAST-NU (QA vs SA nDCG study) |
| Others | South Africa (water QSVM), Uzbekistan (irrigation QUBO), Taiwan (AQI QK-LSTM), Brazil (NASA POWER QNN), Ireland (OpenLitterMap), Portugal (TACO) |

**Benchmark ecosystem**: QuantumCLEF (CLEF 2024/2025, Padua/Milan/Brazil) is the recurring QUBO-FS benchmark lab — worth citing as evidence the subfield is mature.

---

## 10. Open-Source Toolchain (2026 state, verified)

**QUBO side:**
- ⚠️ `neal` is **deprecated** (last release 2022) — never import it. Use `dimod` (0.12.22) + `dwave-samplers` (1.8.0: `SimulatedAnnealingSampler`, `TabuSearchSampler`) for free local solving; Leap QPU/hybrid via token.
- `dwave-examples/mutual-information-feature-selection` (the MIQUBO recipe) was **archived Dec 2024** — reimplement (~100 lines of dimod), don't fork.
- Shortcut: `dwave-scikit-learn-plugin`'s `SelectFromQuadraticModel` drops into any sklearn Pipeline (v0.2.0, Aug 2025). ⚠️ Milano's qcpolimi repos are **AGPL-3.0** (viral — matters if you distribute code). For publishability, write your own QUBO construction.
- `pip install seleqt` — Mücke's QFS with brute-force to ~30 dims.

**QSVM side:**
- `qiskit-machine-learning` **v0.9.1** (released 2026-08-19; 1,095 stars; IBM + Hartree Centre). ⚠️ API moved: v0.8 moved imports under `qiskit_machine_learning.*`; v0.9 removed V1 primitives, requires Qiskit 2.x + Python ≥3.10. **Any tutorial older than ~Nov 2024 uses dead APIs and will not run.** Official tutorial 03 (Quantum Kernel + QSVC) is verified runnable on qiskit 2.5.2 / QML 0.9.1.
- Closest architectural analogues on GitHub: `sebasmos/qml-medimage` (deep embeddings → QSVM, 11 qubits, needs cuQuantum beyond a few thousand samples), `Vatsal-jpg/QABS-SVM` (quantum-inspired band selection → SVM, 91.67% Pavia University). Neither does QUBO-FS or waste data.

**Risk register for the build:**
1. Pin `qiskit~=2.x`, `qiskit-machine-learning==0.9.x`; avoid `BlueprintCircuit` patterns.
2. Never `import neal`; use `dwave.samplers`.
3. D-Wave free-tier minutes exhaust fast — **develop on simulated annealing, reserve QPU for final EXP D numbers**.
4. Cache kernel matrices or subsample — O(n²) in samples at 8 qubits.
5. Verify exactly-8 selection (soft constraints leak; Mücke's α-sweep guarantees it).
6. TACO: state snapshot, count your images, record licenses when redistributing derived CSVs (TACO CC BY 4.0 — attribute; TrashBox has no license — avoid redistribution).

---

## 11. Recommended Framing for the Report/Paper

1. **Claim**: first combination of QUBO feature selection + QSVM on environmental imagery; first litter-specific Pollution Severity Index; first quantum+GIS waste analytics chain. All three verified empty as of 2026-08-25.
2. **Cite as nearest neighbors**: Kumsetty 2022 (quantum TL on TrashBox, India), Nau 2025 (MI-QUBO on MedMNIST), Felefly 2023 (QUBO→QNN on MRI), Mücke 2022 (QFS), SIGIR22 (MIQUBO ablation template), Havlíček 2019 + Schuld 2021 (kernel theory), Huang 2021 (the classical-matches-quantum caveat).
3. **Expect and report honestly**: QUBO-8 ≳ MI-8 on correlated features (your predicted positive result); QSVM ≈ RBF-SVM ±1 point (parity); quantum solver ≈ SA on the same QUBO (parity). Contribution = methodology + first-of-kind application + the ablation discipline itself.
4. **Mandatory extra baseline**: classical RBF-SVM on the same 8 features (reviewers will demand it given Huang 2021).
5. **Metrics**: macro-averaged (TACO imbalance ~667:1); repeated splits with identical budgets across arms; report the QUBO solver, α value, and realized feature count.

---

## 12. Verification Notes (what the critic confirmed/corrected)

- ✅ Villalba-Ferreiro QSVC numbers confirmed against full text (IJCNN 2025).
- ✅ TACO 1,500/4,784 confirmed against paper full text; "9,500 images" claim **refuted**.
- ✅ Strauss & Sharma oil-spill quantum-assisted SVM (IoU 0.60, bal. acc. 0.89) confirmed real.
- ✅ TrashBox 17,785/7-class counts sum exactly; FRUCT'31 venue confirmed; no license (gotcha flagged).
- ✅ Biermann 2020 citation **corrected** (real title: "Finding plastic patches in coastal waters using optical satellite data," Sci Rep 10:5364; Plymouth Marine Lab + Univ. Aegean, not EC JRC).
- ✅ Gwalior Swachh Survekshan total 10,995/12,500 confirmed; component split **does not reconcile** — cite total only.
- ✅ Negative results re-checked independently: no arXiv quantum-waste work; nearest 2025-26 QUBO-FS hits are credit-risk and a separate MNIST QNN.

**Remaining unresolvable gaps**: tacodataset.org down (license/current totals unverified — check at download); both 2022 quantum-trash papers paywalled (only TrashBox FRUCT paper has green OA); non-English venues (Scopus/CNKI) not covered; D-Wave Leap free-tier limits and 8-qubit ZZFeatureMap runtime on IBM open plan must be measured empirically; Nau 2025 peer-review status and the exact Clean Coast Index constant unverified.

---

## Appendix: Raw Data

Per-dimension structured findings (222 findings with URLs, key numbers, and origins):
- [dim_qubo-feature-selection.json](_raw/dim_qubo-feature-selection.json) — 26 findings
- [dim_qsvm-quantum-kernels.json](_raw/dim_qsvm-quantum-kernels.json) — 23
- [dim_taco-dataset.json](_raw/dim_taco-dataset.json) — 22
- [dim_classical-waste-ml.json](_raw/dim_classical-waste-ml.json) — 38
- [dim_quantum-sustainability.json](_raw/dim_quantum-sustainability.json) — 37
- [dim_pollution-index-gis.json](_raw/dim_pollution-index-gis.json) — 16
- [dim_global-qml-landscape.json](_raw/dim_global-qml-landscape.json) — 42
- [dim_open-source-code.json](_raw/dim_open-source-code.json) — 17
- [critique.json](_raw/critique.json) — adversarial verification pass
