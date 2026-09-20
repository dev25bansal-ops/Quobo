"""Unit tests for src.quobo.uncertainty: Wilson intervals and vote agreement."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_wilson_interval_known_values():
    from src.quobo.uncertainty import wilson_interval

    lo, hi = wilson_interval(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert wilson_interval(0, 5) == (0.0, pytest.approx(0.4345, abs=1e-4))
    lo1, hi1 = wilson_interval(5, 5)
    assert lo1 == pytest.approx(0.5655, abs=1e-4) and hi1 == pytest.approx(1.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_interval(3, 0)


def test_wilson_interval_narrows_with_n():
    from src.quobo.uncertainty import wilson_interval

    lo_small, hi_small = wilson_interval(5, 10)
    lo_big, hi_big = wilson_interval(50, 100)
    # same proportion, 10x the evidence -> strictly narrower interval
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_vote_agreement():
    from src.quobo.uncertainty import vote_agreement

    assert vote_agreement([]) == 0.0
    # unanimous (1.0) + 13/25 split (0.52) -> mean 0.76
    assert vote_agreement([{"cigarette": 25}, {"cup": 13, "lid": 12}]) == pytest.approx(0.76)
    # zero-vote crops are skipped, not counted as 0
    assert vote_agreement([{}, {"bottle": 4}]) == pytest.approx(1.0)
