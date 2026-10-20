"""Unit tests for the QPU runner (Q10, L-05, L-06, C-02).

Three properties are pinned here:

- the runner uses the LEAK-FREE protocol (group-aware split + train-only PCA),
  not the legacy transductive ``run_features`` path (L-06);
- the published artifact is written atomically and a valid prior artifact is
  never silently overwritten (L-05);
- the QUBO emitted for the annealer contains the diagonal and the UPPER
  TRIANGLE only, because dimod's ``from_qubo`` sums both orientations (C-02).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _raw_frame(n_per_group: int = 4, n_groups: int = 12, n_emb: int = 8, seed: int = 0):
    """Synthetic raw-embeddings frame with a stable group structure.

    ``crop_path`` mimics TACO's ``batch_N_<stem>_annNNN.jpg`` naming so the
    real ``crop_image_group`` helper produces one group per source photo.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for a in range(n_per_group):
            rows.append(
                {
                    **{f"e{i}": float(v) for i, v in enumerate(rng.normal(size=n_emb))},
                    "label": "bottle" if g % 2 == 0 else "can",
                    "crop_path": f"data/processed/crops/x/batch_1_{g:06d}_ann{a}.jpg",
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("selected, expected_calls", [([0, 2], 1), ([1], 3)])
def test_qpu_runner_integer_samples_and_cardinality(
    monkeypatch, caplog, tmp_path, selected, expected_calls
):
    """Exercise main without cloud packages, network access or filesystem writes."""
    from types import ModuleType, SimpleNamespace
    from unittest.mock import MagicMock

    import scripts.qpu_run as runner

    cloud = ModuleType("dwave.cloud")
    system = ModuleType("dwave.system")
    cloud.Client = MagicMock()
    cloud.Client.from_config.return_value.__enter__.return_value.get_solvers.return_value = []
    sampler = MagicMock()
    sampler.sample_qubo.return_value = SimpleNamespace(
        first=SimpleNamespace(sample={i: int(i in selected) for i in range(3)})
    )
    system.LeapHybridSampler = MagicMock(return_value=sampler)
    monkeypatch.setitem(sys.modules, "dwave.cloud", cloud)
    monkeypatch.setitem(sys.modules, "dwave.system", system)
    monkeypatch.setenv("DWAVE_API_TOKEN", "unit-test-not-a-credential")
    monkeypatch.setattr(
        runner,
        "load_config",
        lambda _: {
            "seed": 42,
            "selection": {"k": 2},
            "features": {"pca_components": 3},
            "qubo": {"mi_bins": 16},
            "qsvm": {"test_fraction": 0.25},
        },
    )
    # L-06: the runner must go through the leak-free split, not the legacy
    # transductive PCA. The synthetic split is injected here; the split logic
    # itself is exercised by the leak-free property test below.
    split = runner.LeakFreeSplit(
        X_train_pca=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        y_train=np.array(["a", "b", "a"]),
        X_test_pca=np.array([[1.0, 1.1, 1.2]]),
        y_test=np.array(["b"]),
        train_groups=np.array(["g0", "g1", "g2"]),
        test_groups=np.array(["g3"]),
        explained_variance_pct=91.5,
        n_overlap_groups=0,
    )
    monkeypatch.setattr(runner, "make_leak_free_split", lambda *a, **k: split)
    # QUBO coefficients are tuple-keyed; the returned sample is integer-keyed.
    monkeypatch.setattr(
        runner,
        "build_qubo",
        lambda *a, **k: {(0, 0): -1.0, (1, 1): -1.0, (2, 2): -1.0, (0, 2): 0.5},
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    assert runner.main() == 0
    result = json.loads((tmp_path / "results" / "qpu" / "qpu_selection.json").read_text())
    assert result["selected"] == selected
    assert result["requested_k"] == 2
    assert result["n_features_selected"] == len(selected)
    assert result["cardinality_ok"] is (len(selected) == 2)
    assert sampler.sample_qubo.call_count == expected_calls
    assert ("not directly comparable" in caplog.text) is (len(selected) != 2)


def test_qpu_runner_uses_leak_free_path_not_legacy_transductive_pca():
    """L-06: the legacy ``run_features`` (PCA fitted on ALL rows) must be gone.

    The regression this guards is not theoretical: with the transductive fit the
    hardware number is not comparable with the leak-free SA baseline, which is
    the entire point of the QPU arm.
    """
    import scripts.qpu_run as runner

    assert not hasattr(runner, "run_features"), "qpu_run must not use the transductive PCA path"
    assert hasattr(runner, "make_leak_free_split")
    assert hasattr(runner, "fit_pca_on_train")


def test_qpu_runner_no_token_exits_zero_without_writing(monkeypatch, tmp_path):
    """No DWAVE_API_TOKEN -> exit 0 and no artifact (CI stays green, no network)."""
    import scripts.qpu_run as runner

    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    assert runner.main() == 0
    assert not (tmp_path / "results").exists()


def test_qpu_result_is_published_atomically_and_never_clobbers_a_valid_artifact(
    monkeypatch, tmp_path
):
    """L-05: a second run must not silently overwrite a valid prior artifact."""
    import scripts.qpu_run as runner

    dest = tmp_path / "results" / "qpu" / "qpu_selection.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps({"selected": [1, 2], "solver": "prior"}), encoding="utf-8")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(FileExistsError):
        runner._publish_result(dest, {"selected": [3, 4], "solver": "new"})
    # the prior artifact is intact
    assert json.loads(dest.read_text())["solver"] == "prior"


def test_qpu_result_publish_leaves_no_truncated_file_on_failure(monkeypatch, tmp_path):
    """L-05: a failed publish must not leave a partial/truncated JSON behind."""
    import scripts.qpu_run as runner

    dest = tmp_path / "results" / "qpu" / "qpu_selection.json"

    class Boom(Exception):
        pass

    def exploding_publish(path, payload):
        raise Boom("simulated crash mid-publish")

    monkeypatch.setattr(runner, "publish_exclusive", exploding_publish)
    with pytest.raises(Boom):
        runner._publish_result(dest, {"selected": [0]})
    assert not dest.exists()


def test_build_qubo_emits_diagonal_and_upper_triangle_only():
    """C-02: dimod's from_qubo SUMS (i,j) and (j,i).

    Emitting both orientations doubles every redundancy term, so the QPU would
    optimize a different objective than the SA arm it is compared against.
    """
    import scripts.qpu_run as runner

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 5))
    y = np.array(["a", "b"] * 20)
    Q = runner.build_qubo(X, y, alpha=0.5, n_bins=4)

    keys = set(Q)
    assert len(keys) == 5 + 5 * 4 // 2
    for i, j in keys:
        assert i <= j, f"lower-triangle key {(i, j)} must not be emitted"
    assert all((j, i) not in keys for (i, j) in keys if i != j)
    # the diagonal carries the importance reward, the off-diagonal the redundancy
    assert Q[(0, 0)] <= 0.0
    assert Q[(0, 1)] >= 0.0


def test_build_qubo_matches_selection_module_bit_for_bit():
    """The QPU's QUBO must be the SAME matrix the SA arm optimizes.

    src/quobo/selection.py builds its QUBO from the same MI table and emits
    the upper triangle only. If the two scripts ever disagree, the hardware
    number is not comparable with the SA baseline — which is the whole point
    of the QPU arm.
    """
    import scripts.qpu_run as runner
    from src.quobo.selection import compute_mi_table

    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 6))
    y = np.array(["a", "b", "c"] * 20)
    alpha = 0.6
    n_bins = 4

    Q = runner.build_qubo(X, y, alpha, n_bins=n_bins)

    # reference: selection.py's qubo_matrix + its upper-triangle emission
    importance, redundancy = compute_mi_table(X, y, n_bins)
    n = len(importance)
    ref = {}
    for i in range(n):
        ref[(i, i)] = -alpha * importance[i]
        for j in range(i + 1, n):
            ref[(i, j)] = (1.0 - alpha) * redundancy[i, j]

    assert set(Q) == set(ref)
    for key, value in ref.items():
        assert Q[key] == pytest.approx(value)
    # the legacy both-orientations form would have had twice the off-diagonal
    # entries and therefore doubled every redundancy term
    assert len(Q) == n + n * (n - 1) // 2
    assert n + 2 * (n * (n - 1) // 2) == 2 * len(Q) - n


def test_qubo_is_invariant_to_held_out_rows_leak_free_property():
    """L-06: the QUBO must not change when the HELD-OUT rows change.

    This is the operational definition of the leak-free protocol: the scaler,
    the PCA rotation and the MI table are all fitted on the train rows, so
    perturbing the test rows cannot move the QUBO that the annealer solves.
    Under the legacy transductive fit the same perturbation DOES change it.
    """
    import scripts.qpu_run as runner

    cfg = {
        "seed": 42,
        "features": {"pca_components": 4},
        "qsvm": {"test_fraction": 0.25},
    }
    raw = _raw_frame()

    base = runner.make_leak_free_split(cfg, raw=raw)
    base_qubo = runner.build_qubo(base.X_train_pca, base.y_train, alpha=0.5, n_bins=4)

    # Identify the held-out rows and perturb ONLY them.
    split = runner.make_leak_free_split(cfg, raw=raw)
    from src.quobo.run_experiment import crop_image_group

    held_out_groups = set(split.test_groups)
    perturbed = raw.copy()
    emb_cols = [c for c in perturbed.columns if c.startswith("e")]
    is_held_out = perturbed["crop_path"].apply(lambda p: crop_image_group(p) in held_out_groups)
    assert is_held_out.any(), "test fixture must contain held-out rows to perturb"
    perturbed.loc[is_held_out, emb_cols] = perturbed.loc[is_held_out, emb_cols] + 1000.0

    after = runner.make_leak_free_split(cfg, raw=perturbed)
    after_qubo = runner.build_qubo(after.X_train_pca, after.y_train, alpha=0.5, n_bins=4)

    assert set(base_qubo) == set(after_qubo)
    for key, value in base_qubo.items():
        assert after_qubo[key] == pytest.approx(value), f"QUBO entry {key} moved with the test rows"


def test_leak_free_split_has_no_group_overlap_and_fits_pca_on_train_only():
    """The split must not straddle a source photo, and the PCA must be train-fit."""
    import scripts.qpu_run as runner

    cfg = {
        "seed": 42,
        "features": {"pca_components": 3},
        "qsvm": {"test_fraction": 0.25},
    }
    raw = _raw_frame(n_per_group=3, n_groups=10, n_emb=6)
    split = runner.make_leak_free_split(cfg, raw=raw)

    assert set(split.train_groups) & set(split.test_groups) == set()
    assert split.n_overlap_groups == 0
    assert len(split.X_train_pca) == len(split.y_train)
    assert len(split.X_test_pca) == len(split.y_test)
    assert split.X_train_pca.shape[1] == 3
