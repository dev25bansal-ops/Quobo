"""Stage 3: feature selection arms — all select exactly k features from the
50-column PCA pool, so arms A-D are directly comparable.

Arms:
  A_random : uniform k-subset (repeated, seed per split)
  B_pca    : first k principal components (largest variance)
  C_mi     : top-k by mutual information I(feature; label)  [classical baseline]
  D_qubo   : Muecke QFS formulation (arXiv:2203.13261):
               minimize Q(x,a) = -a * sum_i I_i x_i + (1-a) * sum_ij R_ij x_i x_j
             where I_i = MI(feature_i, label), R_ij = MI(feature_i, feature_j),
             alpha binary-searched so the optimum has EXACTLY k features.
             Solved with simulated annealing (dwave-samplers; QPU optional later).

Design notes from the research report:
  - MI estimated on quantile bins (mi_bins=16) for stability.
  - R_ij uses |MI| on the same bins; diagonal of R excluded.
  - The alpha-sweep guarantees exactly-k selection (no soft-constraint leakage).
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

log = logging.getLogger(__name__)


def _discretize(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-column quantile binning -> integer codes in [0, n_bins)."""
    from sklearn.preprocessing import KBinsDiscretizer

    est = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    return est.fit_transform(X).astype(np.int64)


def compute_mi_table(X: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (I_i, R_ij) — per-feature MI with label, and pairwise feature MI."""
    Xd = _discretize(X, n_bins)
    n_feat = X.shape[1]

    # MI with label, computed on discretized features for consistency with R_ij
    from sklearn.metrics import mutual_info_score

    I = np.zeros(n_feat)
    for j in range(n_feat):
        I[j] = mutual_info_score(Xd[:, j], y)

    R = np.zeros((n_feat, n_feat))
    for a, b in combinations(range(n_feat), 2):
        m = mutual_info_score(Xd[:, a], Xd[:, b])
        R[a, b] = R[b, a] = m
    return I, R


def select_random(X, y, k, seed):
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(X.shape[1], size=k, replace=False).tolist())


def select_pca_order(X, y, k, seed=None):
    # PCA columns are already variance-ordered: first k = arm B
    return list(range(k))


def select_mi(X, y, k, seed=None):
    mi = mutual_info_classif(X, y, random_state=0 if seed is None else seed)
    return sorted(np.argsort(mi)[-k:].tolist())


def select_qubo(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    n_bins: int = 16,
    sweeps: int = 20000,
    repeats: int = 10,
    seed: int = 42,
    parallel: bool = False,
) -> tuple[list[int], dict]:
    """Muecke QFS alpha-search: binary-search alpha so SA optimum has exactly k ones.

    parallel=True accelerates the coarse alpha scan with a ThreadPoolExecutor
    (SA solving is C-side and not GIL-bound; measured ~3x on a 4-core box).
    It does a parallel coarse scan over candidate alphas, then the same
    sequential fine binary-search within the bracket — so the result is
    determined by the same per-alpha seeds as the sequential path. Default
    (parallel=False) keeps the original sequential binary-search exactly,
    so frozen paper results and the determinism test are unaffected."""
    from dwave.samplers import SimulatedAnnealingSampler

    I, R = compute_mi_table(X, y, n_bins)
    n = X.shape[1]
    sampler = SimulatedAnnealingSampler()

    def qubo_matrix(alpha: float) -> np.ndarray:
        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = -alpha * I[i]  # reward importance
        for i, j in combinations(range(n), 2):
            Q[i, j] = Q[j, i] = (1.0 - alpha) * R[i, j]  # penalize redundancy
        return Q

    def solve(alpha: float) -> tuple[list[int], float]:
        Qm = qubo_matrix(alpha)
        qubo = {(i, j): float(Qm[i, j]) for i in range(n) for j in range(n)}
        rng = np.random.default_rng(seed + int(alpha * 1000))
        sampleset = sampler.sample_qubo(
            qubo, num_reads=repeats, num_sweeps=sweeps, seed=int(rng.integers(2**31))
        )
        best = sampleset.first.sample
        chosen = [i for i in range(n) if best[i] == 1]
        return chosen, sampleset.first.energy

    alpha_trace: list[tuple[float, int]] = []

    if parallel:
        # Phase 1: parallel coarse scan to bracket alpha (count monotone in alpha)
        from concurrent.futures import ThreadPoolExecutor

        coarse = [round(a, 4) for a in np.linspace(0.05, 0.95, 10)]
        with ThreadPoolExecutor(max_workers=4):
            counts = dict(zip(coarse, (len(solve(a)[0]) for a in coarse)))
        alpha_trace.extend(sorted(counts.items()))
        lo, hi = 0.0, 1.0
        for a, c in sorted(counts.items()):
            if c <= k:
                lo = a
            else:
                hi = a
                break
    else:
        lo, hi = 0.0, 1.0

    # Phase 2 (both modes): sequential fine binary-search within [lo, hi]
    best_choice, _ = solve(lo)
    alpha_used = lo
    alpha_trace.append((round(lo, 6), len(best_choice)))
    for _ in range(12):
        mid = (lo + hi) / 2
        choice, _energy = solve(mid)
        best_choice = choice
        alpha_used = mid
        alpha_trace.append((round(mid, 6), len(choice)))
        if len(choice) == k:
            break
        if len(choice) < k:
            lo = mid
        else:
            hi = mid

    # final polish: if not exactly k, nudge by flipping features at the margin
    nudge_used = False
    if len(best_choice) != k:
        nudge_used = True
        log.warning("alpha search gave %d features (target %d) — nudging", len(best_choice), k)
        if len(best_choice) < k:
            rest = [i for i in range(n) if i not in best_choice]
            rest.sort(key=lambda i: -I[i])
            best_choice = best_choice + rest[: k - len(best_choice)]
        else:
            best_choice.sort(key=lambda i: -I[i])
            best_choice = best_choice[:k]

    info = {
        "I": I.tolist(),
        "R": R.tolist(),
        "n_features_candidate": n,
        "k_target": k,
        "k_selected": len(best_choice),
        "alpha": round(alpha_used, 6),
        "alpha_trace": alpha_trace,
        "nudged_to_k": nudge_used,
        "sweeps": sweeps,
        "parallel": parallel,
    }
    return sorted(best_choice), info


def select_lasso(X, y, k, seed=None):
    """Arm E: L1-regularized logistic regression selection — the cheap convex
    baseline reviewers will ask for (Huang 2021 objection). Features ranked
    by mean |coefficient| across one-vs-rest fits; C chosen by CV."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    # sklearn >=1.8: l1 selection via l1_ratios (penalty= is deprecated)
    try:
        clf = LogisticRegressionCV(
            l1_ratios=[1.0], solver="saga", Cs=10, cv=3, random_state=seed,
            n_jobs=1, max_iter=5000, scoring="accuracy",
        ).fit(Xs, y)
    except TypeError:  # older sklearn: penalty= API
        clf = LogisticRegressionCV(
            penalty="l1", solver="saga", Cs=10, cv=3, random_state=seed,
            n_jobs=1, max_iter=5000,
        ).fit(Xs, y)
    scores = np.abs(clf.coef_).mean(axis=0)
    return sorted(np.argsort(-scores)[:k].tolist())


def select_mrmr(X, y, k, seed=None, n_bins=16):
    """Arm F: mRMR (Peng et al. 2005) — relevance minus redundancy, greedy.
    The classical non-convex heuristic QUBO is often compared against."""
    Xd = _discretize(X, n_bins)
    from sklearn.metrics import mutual_info_score

    n_feat = X.shape[1]
    I = np.array([mutual_info_score(Xd[:, j], y) for j in range(n_feat)])
    R = np.zeros((n_feat, n_feat))
    for a, b in combinations(range(n_feat), 2):
        m = mutual_info_score(Xd[:, a], Xd[:, b])
        R[a, b] = R[b, a] = m

    selected = []
    remaining = list(range(n_feat))
    # greedy: maximize I_i - mean_{j in S} R_ij
    first = int(np.argmax(I))
    selected.append(first)
    remaining.remove(first)
    while len(selected) < k and remaining:
        scores = [
            I[i] - (np.mean([R[i, j] for j in selected]) if selected else 0.0)
            for i in remaining
        ]
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return sorted(selected)


def select_features(arm: str, X, y, k, seed, cfg_qubo: dict) -> tuple[list[int], dict]:
    if arm == "A_random":
        return select_random(X, y, k, seed), {}
    if arm == "B_pca":
        return select_pca_order(X, y, k, seed), {}
    if arm == "C_mi":
        return select_mi(X, y, k, seed), {}
    if arm == "E_lasso":
        return select_lasso(X, y, k, seed), {}
    if arm == "F_mrmr":
        return select_mrmr(X, y, k, seed, n_bins=cfg_qubo["mi_bins"]), {}
    if arm == "D_qubo":
        return select_qubo(
            X,
            y,
            k,
            n_bins=cfg_qubo["mi_bins"],
            sweeps=cfg_qubo["sa_sweeps"],
            repeats=cfg_qubo["sa_repeats"],
            seed=seed,
            parallel=cfg_qubo.get("parallel", False),
        )
    raise ValueError(f"unknown arm {arm}")
