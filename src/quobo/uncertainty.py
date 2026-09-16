"""A1: uncertainty-aware PSI — propagate classifier uncertainty into zone indices.

The 25-repeat majority vote already stores a per-crop vote distribution; that
distribution IS the uncertainty. Two layers:

1. Per-crop vote agreement (0-1): fraction of repeats agreeing with the
   majority label. A crop that splits 13/12 across repeats is noise, and
   pretending otherwise inflates zone-level confidence.
2. Per-zone Wilson score intervals on the 'dirty' rate (the PSI driver):
   the reported PSI band inherits the worst-case (upper) bound, so a zone
   only reaches 'Hazardous' when the *lower* bound of its evidence supports it.

Wilson is used over naive +/-1.96*sqrt(p(1-p)/n) because it keeps intervals
inside [0,1] for small n and never produces the degenerate zero-width
interval at p=0 or p=1 that the normal approximation does.

References: Wilson 1927 (JASA 22:209); Brown, Cai & DasGupta 2001
(Int. Statist. Rev. 69:101) recommend it for binomial proportions at small n.
"""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (lo, hi).

    Examples
    --------
    >>> lo, hi = wilson_interval(8, 10)
    >>> 0.48 < lo < 0.50 and 0.88 < hi < 0.90
    True
    >>> wilson_interval(0, 5)
    (0.0, 0.0)
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = max(0.0, (center - spread) / denom)
    hi = min(1.0, (center + spread) / denom)
    return (lo, hi)


def vote_agreement(votes_per_crop: list[dict[str, int]]) -> float:
    """Mean majority-vote agreement across crops, in [0,1].

    votes_per_crop: one {class_label: count} dict per crop (from the 25
    per-rep predictions). Agreement = majority count / total votes, averaged
    over crops with at least one vote.
    """
    agreements = []
    for counts in votes_per_crop:
        total = sum(counts.values())
        if total == 0:
            continue
        agreements.append(max(counts.values()) / total)
    if not agreements:
        return 0.0
    return sum(agreements) / len(agreements)
