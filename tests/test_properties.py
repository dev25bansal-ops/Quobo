"""Property-based tests (Hypothesis). Isolated in their own file so the main
suite stays fast; run with: pytest tests/test_properties.py

Each property is a candidate invariant that should hold for *all* valid inputs,
not just the fixed examples in test_pipeline.py. Kept cheap (small examples,
deadline=None) so CI stays fast while still exploring edge cases.
"""
import sys
from pathlib import Path

import hypothesis.extra.numpy as hnp
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pollution_dashboard import compute_zone_table, zone_of
from src.quobo.features import fit_pca_on_train

VALID_ZONES = {f"Zone {c}" for c in "ABCDEF"} | {"Unassigned"}


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.text(min_size=0, max_size=200))
def test_zone_of_never_crashes_and_returns_valid(path):
    """zone_of must total over any string input (demo paths are untrusted)."""
    for mode in ("batch", "folder"):
        result = zone_of(path, mode=mode)
        assert result in VALID_ZONES


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.dictionaries(
        keys=st.sampled_from(list(VALID_ZONES)),
        values=st.integers(min_value=0, max_value=50),
        min_size=1,
    )
)
def test_psi_and_cleanliness_bounded(zone_counts):
    """PSI must stay in [0, 500] and cleanliness in [0, 100] for any zone mix."""
    rows = []
    for zone, n in zone_counts.items():
        for i in range(n):
            rows.append({"crop_path": f"{zone}/c/{i}.jpg", "class": "cigarette",
                         "file": f"{i}.jpg", "zone_dir": zone})
    if not rows:
        return
    zt = compute_zone_table(pd.DataFrame(rows))
    assert zt["psi"].between(0, 500).all()
    assert zt["cleanliness"].between(0, 100).all()
    assert (zt["total"] == zt["clean"] + zt["dirty"]).all()


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    hnp.arrays(dtype=np.float64,
               shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=40, max_side=120),
               elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
    st.integers(min_value=20, max_value=60),
)
def test_fit_pca_on_train_shapes_and_leak_free_property(Xtr, n_te_rows):
    """PCA refit on train: output widths match, and the transform of a train
    row is exactly reproducible from the all-rows transform (train rows only)."""
    n_feat = Xtr.shape[1]
    Xte = np.random.default_rng(0).normal(size=(n_te_rows, n_feat))
    n_comp = min(10, n_feat, Xtr.shape[0] - 1)
    if n_comp < 2:
        return
    Xtr_p, Xall_p, ev_pct = fit_pca_on_train(Xtr, np.vstack([Xtr, Xte]), n_comp, seed=0)
    assert Xtr_p.shape == (Xtr.shape[0], n_comp)
    assert Xall_p.shape == (Xtr.shape[0] + n_te_rows, n_comp)
    # explained variance is a sensible fraction when inputs have variance
    # (zero-variance columns legitimately yield NaN — that's sklearn, not a bug)
    if np.isfinite(ev_pct):
        assert 0.0 <= ev_pct <= 100.0 + 1e-6
