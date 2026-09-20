---
tags: [research, results]
up: "[[00 Home]]"
---

# Key Results (leak-free v2)

Definitive numbers: 6-class TACO, 8-feature budget, 25 group-aware repeats,
paired significance with Holm correction. Source: `results/tables/`,
`research/ISSUE_REGISTER.md` §Final state.

| Arm | Accuracy | vs QUBO-8 |
|---|---|---|
| Full-50 (ceiling) | **48.3%** | — |
| LASSO-8 | 41.4% | tie |
| mRMR-8 | 41.4% | tie |
| **QUBO-8** | **40.9%** | — |
| PCA-8 | 40.1% | tie (p>0.5) |
| MI-8 | 39.5% | tie |
| Random-8 | 32.2% | **p<0.001 worse** |
| QSVC | ~37% | kernel parity with classical |

## Interpretation

1. All principled selection arms **statistically tie** under the strict protocol;
   only random selection is significantly worse.
2. QUBO's contribution is not accuracy but a **globally-optimal formulation**
   (MI relevance − redundancy) that runs unchanged on quantum annealers —
   motivation for [[quantum_experiments]] and the S3 QPU item ([[Issue Register]]).
3. Full-50 > any 8-subset: 50→8 compression costs ~7 pts; selection can't recover
   MobileNetV2 information loss.
4. v1 (transductive PCA) inflated all arms by ~1–2 pts; v1-vs-v2 materiality is
   reported transparently in the paper.

## Artifacts

- `results/tables/significance_*.csv` (78 comparisons)
- `results/figures/` (confusion matrices via `scripts/confusion_analysis.py`)
- `results/dashboard/` (PSI zone analytics — see PSI disclaimer in [[Scripts]])
- `results/onnx/` (edge bundle, 100% parity vs sklearn)
