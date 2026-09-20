"""Conditional PSI uncertainty uses class rates, including clean-only zones."""

import math
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pollution_dashboard import HAZARD_W, compute_zone_table, render_dashboard


def test_one_bottle_interval_matches_hand_calculation(tmp_path):
    frame = pd.DataFrame([{"crop_path": "batch_1_one.jpg", "class": "bottle"}])
    row = compute_zone_table(frame).iloc[0]
    # Six Bonferroni-adjusted Wilson intervals, n=1. For 1/1 bottle:
    # lower=1/(1+z²); absent classes have lower=0. Observed reference=1.
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(HAZARD_W)))
    expected_lo = math.floor(500 / (1 + z * z))
    assert row["psi"] == 500
    assert row["psi_lo"] == expected_lo
    assert row["psi_hi"] == 500
    assert row["psi_ci"] == f"[{expected_lo}-500]"
    assert row["dirty_rate_lo"] == pytest.approx(0)
    assert row["dirty_rate_hi"] > 0
    out = tmp_path / "dashboard.html"
    render_dashboard(pd.DataFrame([row]), out)
    html = out.read_text(encoding="utf-8")
    assert "conditional" in html.lower()
    assert "calibrated" in html.lower()


def test_unknown_and_empty_zones_have_explicit_behavior():
    empty = compute_zone_table(pd.DataFrame(columns=["crop_path", "class"]))
    assert {"zone", "psi", "psi_lo", "psi_hi", "psi_ci"} <= set(empty.columns)
    with pytest.raises(ValueError, match="class"):
        compute_zone_table(pd.DataFrame([{"crop_path": "x", "class": "unknown"}]))
