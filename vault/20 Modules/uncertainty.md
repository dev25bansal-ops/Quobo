---
tags: [module]
path: src/quobo/uncertainty.py
up: "[[00 Home]]"
---

# uncertainty

Small statistics helper consumed by `scripts/pollution_dashboard.py`.

- `wilson_interval(k, n, z)` — Wilson score interval for per-zone cleanliness
  proportions (robust for small n, unlike normal-approx).
- `vote_agreement(preds_per_rep)` — agreement across repeats (how consistent the
  model's per-crop label is over the ensemble of runs); feeds the dashboard's
  confidence display.

Related: [[Scripts]] · [[Key Results]]
