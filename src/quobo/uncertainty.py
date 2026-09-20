"""Binomial sampling intervals and descriptive prediction stability.

Wilson intervals assume independent Bernoulli observations. Repeated model
votes and crops from the same image are not independent validation samples;
vote agreement is not a calibrated probability of correctness.
"""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval; no observations give [0, 1].

    >>> lo, hi = wilson_interval(8, 10)
    >>> 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    True
    >>> lo, hi = wilson_interval(0, 5)
    >>> lo == 0.0 and 0.43 < hi < 0.44
    True
    """
    if not isinstance(successes, int) or not isinstance(n, int):
        raise TypeError("successes and n must be integers")
    if isinstance(successes, bool) or isinstance(n, bool):
        raise TypeError("successes and n must not be booleans")
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("require 0 <= successes <= n")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = max(0.0, (center - spread) / denom)
    hi = min(1.0, (center + spread) / denom)
    return (lo, hi)


def vote_agreement(votes_per_crop: list[dict[str, int]]) -> float:
    """Mean majority fraction over crops with votes; empty input returns zero."""
    agreements = []
    for counts in votes_per_crop:
        if any(not isinstance(v, int) or isinstance(v, bool) for v in counts.values()):
            raise TypeError("vote counts must be integers")
        if any(v < 0 for v in counts.values()):
            raise ValueError("vote counts must be nonnegative")
        total = sum(counts.values())
        if total == 0:
            continue
        agreements.append(max(counts.values()) / total)
    if not agreements:
        return 0.0
    return sum(agreements) / len(agreements)
