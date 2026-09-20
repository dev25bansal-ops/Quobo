"""Unit tests for run_experiment: Holm-corrected significance, early stopping,
and MLflow tracking (opt-in, graceful)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quobo.run_experiment import paired_significance


# ---------- Holm correction ----------


def test_holm_monotone_and_bounded():
    # replicate the holm() inside paired_significance via its output shape:
    # feed a results frame with two arms x one classifier, 25 reps
    rng = np.random.default_rng(3)
    rows = []
    for rep in range(25):
        rows.append(
            {
                "repeat": rep,
                "classifier": "rbf_svm",
                "arm": "A",
                "accuracy": 0.40 + 0.02 * rng.normal(),
            }
        )
        rows.append(
            {
                "repeat": rep,
                "classifier": "rbf_svm",
                "arm": "B",
                "accuracy": 0.40 + 0.02 * rng.normal(),
            }
        )
    out = paired_significance(pd.DataFrame(rows))
    assert len(out) == 1
    assert 0.0 <= out[0]["wilcoxon_p_holm"] <= 1.0
    assert isinstance(out[0]["significant_0.05"], bool)


def test_paired_t_holm_matches_independent_holm():
    """Check paired-t adjustment against a vectorized Holm reference.

    The public result rounds p-values to four decimal places.
    """
    from scipy.stats import ttest_rel

    rng = np.random.default_rng(11)
    arms = ["A", "B", "C", "D"]
    rows = []
    # B systematically above A -> small raw t p-value for (A,B) family member
    for rep in range(30):
        base = 0.40 + 0.02 * rng.normal()
        rows.append({"repeat": rep, "classifier": "rbf_svm", "arm": "A", "accuracy": base})
        rows.append(
            {
                "repeat": rep,
                "classifier": "rbf_svm",
                "arm": "B",
                "accuracy": base + 0.03 + 0.01 * rng.normal(),
            }
        )
        rows.append(
            {
                "repeat": rep,
                "classifier": "rbf_svm",
                "arm": "C",
                "accuracy": base + 0.02 * rng.normal(),
            }
        )
        rows.append(
            {
                "repeat": rep,
                "classifier": "rbf_svm",
                "arm": "D",
                "accuracy": base - 0.02 * rng.normal(),
            }
        )
    out = paired_significance(pd.DataFrame(rows))
    assert len(out) == 6  # C(4,2) comparisons, one classifier

    wide = pd.DataFrame(rows).pivot_table(index="repeat", columns="arm", values="accuracy")
    raw_t = []
    for a, b in [(x, y) for i, x in enumerate(arms) for y in arms[i + 1 :]]:
        diff = (wide[a] - wide[b]).dropna()
        _, tp = ttest_rel(diff, np.zeros_like(diff))
        raw_t.append(float(tp))
    raw_t = np.asarray(raw_t)
    order = np.argsort(raw_t)
    expected = np.empty_like(raw_t)
    expected[order] = np.minimum(
        1.0, np.maximum.accumulate(raw_t[order] * np.arange(len(raw_t), 0, -1))
    )
    got = [r["paired_t_p_holm"] for r in out]
    rounded = [round(float(p), 4) for p in expected]
    assert got == rounded
    # Ensure this fixture catches the original raw-p-value passthrough bug.
    assert rounded != [round(float(p), 4) for p in raw_t]
    assert all(g >= r - 0.00005 for g, r in zip(got, raw_t))


@pytest.mark.parametrize("values", [[0.4], [0.4, 0.5, 0.6]])
def test_paired_t_identical_or_insufficient_pairs(values):
    rows = [
        {"repeat": rep, "classifier": "rbf_svm", "arm": arm, "accuracy": value}
        for rep, value in enumerate(values)
        for arm in ("A", "B")
    ]
    result = paired_significance(pd.DataFrame(rows))[0]
    assert result["paired_t_p_holm"] == 1.0
    assert result["wilcoxon_p_holm"] == 1.0
    assert result["significant_0.05"] is False


# ---------- Enhancement 3: early stopping ----------


def test_should_stop_early_flat():
    from src.quobo.run_experiment import should_stop_early

    rows = []
    for rep in range(15):
        rows.append(
            {
                "repeat": rep,
                "arm_key": "D_qubo",
                "classifier": "rbf_svm",
                "accuracy": 0.41 + (rep % 2) * 0.001,
            }
        )  # flat within 0.001
    assert should_stop_early(rows, None, min_reps=10, window=5, tol=0.01) is not None


def test_should_stop_early_not_flat():
    from src.quobo.run_experiment import should_stop_early

    rows = []
    for rep in range(15):
        rows.append(
            {
                "repeat": rep,
                "arm_key": "D_qubo",
                "classifier": "rbf_svm",
                "accuracy": 0.30 + rep * 0.01,
            }
        )  # clearly rising
    assert should_stop_early(rows, None, min_reps=10, window=5, tol=0.01) is None


def test_should_stop_early_needs_min_reps():
    from src.quobo.run_experiment import should_stop_early

    rows = [
        {"repeat": rep, "arm_key": "D_qubo", "classifier": "rbf_svm", "accuracy": 0.41}
        for rep in range(5)
    ]  # < min_reps
    assert should_stop_early(rows, None, min_reps=10, window=5, tol=0.01) is None


# ---------- Enhancement 10: MLflow tracking (opt-in, graceful) ----------


def test_log_to_mlflow_graceful_and_valid(tmp_path):
    """log_to_mlflow logs when mlflow is present (sqlite store), and degrades
    to False (never raises) when it is absent."""
    from src.quobo.run_experiment import log_to_mlflow

    cfg = {
        "seed": 42,
        "data": {"classes": ["a", "b"], "min_images_per_class": 40},
        "features": {"backbone": "MobileNetV2", "pca_components": 50},
        "selection": {"k": 8},
        "qubo": {"mi_bins": 16, "sa_sweeps": 20000, "sa_repeats": 10},
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 400,
            "test_fraction": 0.25,
            "tune_rbf": True,
        },
        "experiment": {"n_repeats": 25, "arms": ["A_random", "D_qubo"], "track_mlflow": True},
    }
    res = pd.DataFrame(
        [
            {
                "repeat": 0,
                "arm_key": "D_qubo",
                "arm": "QUBO-8",
                "classifier": "rbf_svm",
                "accuracy": 0.40,
                "macro_f1": 0.38,
            },
            {
                "repeat": 0,
                "arm_key": "A_random",
                "arm": "Random-8",
                "classifier": "rbf_svm",
                "accuracy": 0.32,
                "macro_f1": 0.30,
            },
        ]
    )
    res.to_csv(tmp_path / "results_all.csv", index=False)
    try:
        import mlflow  # noqa: F401

        have_mlflow = True
    except ImportError:
        have_mlflow = False
    result = log_to_mlflow(res, cfg, tmp_path)
    if have_mlflow:
        assert result is True
    else:
        assert result is False  # graceful degradation, never raises
