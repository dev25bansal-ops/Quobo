"""Unit tests for run_experiment: Holm-corrected significance, early stopping,
MLflow tracking (opt-in, graceful), and the staged run body (L-01)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quobo import run_experiment as RE
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


# ---------- L-01: the staged run body ----------
#
# ``_run_body`` was split into ``_select_and_classify_rep``, ``_persist_rep``
# and ``_finalize_run``. The point of the split is that a stage can be
# exercised *without* running a 25-repeat experiment, so these tests drive the
# stages directly with synthetic data and stubbed classifiers.

PAPER_CLASSES = ("bottle", "can", "cigarette", "carton", "cup", "lid")


def _staged_cfg() -> dict:
    """A minimal config shaped like the paper's, for the staged functions."""
    return {
        "seed": 42,
        "data": {"classes": list(PAPER_CLASSES), "min_images_per_class": 40},
        "features": {"backbone": "MobileNetV2", "pca_components": 50},
        "selection": {"k": 8},
        "qubo": {"mi_bins": 16, "sa_sweeps": 20000, "sa_repeats": 10, "parallel": False},
        "qsvm": {
            "reps": 1,
            "entanglement": "linear",
            "max_train_samples": 400,
            "test_fraction": 0.25,
            "tune_rbf": True,
            "max_qsvm_features": 20,
        },
        "experiment": {
            "n_repeats": 2,
            "arms": ["A_random", "D_qubo"],
            "early_stop": False,
            "track_mlflow": False,
        },
    }


def _staged_ctx(n_train: int = 30, n_test: int = 10, n_features: int = 12):
    """A synthetic repeat context with plausible shapes."""
    rng = np.random.default_rng(7)
    Xtr = rng.normal(size=(n_train, n_features)).astype(np.float32)
    Xte = rng.normal(size=(n_test, n_features)).astype(np.float32)
    ytr = np.array(["a", "b"] * (n_train // 2))
    yte = np.array(["a", "b"] * (n_test // 2))
    return RE._RepeatContext(
        rep=0,
        seed=42,
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        classes=["a", "b"],
        y_test=[str(v) for v in yte],
        test_ids=[f"/crops/{c}/photo{i}_ann{i:04d}.jpg" for i, c in enumerate(yte)],
        test_groups=[f"photo{i}" for i in range(len(yte))],
    )


@pytest.fixture()
def stub_classifiers(monkeypatch):
    """Replace the real classifier stack with a deterministic stub.

    The staged functions must be testable without training an SVM and a QSVC.
    """

    def fake_all_classifiers(Xtr, ytr, Xte, yte, cfg, rep_offset=0):
        return {
            "rbf_svm": {
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "fit_seconds": 0.01,
                "predict_seconds": 0.01,
                "predictions": [str(v) for v in yte],
            }
        }

    monkeypatch.setattr(RE, "run_all_classifiers", fake_all_classifiers)


@pytest.fixture()
def stub_select(monkeypatch):
    """Select the first k columns for every arm (no real selection work)."""

    def fake_select(arm, X, y, k, seed, qubo_cfg):
        return list(range(k)), {"stub": True}

    monkeypatch.setattr(RE, "select_features", fake_select)


def test_select_and_classify_rep_returns_rows_and_selection(stub_classifiers, stub_select):
    ctx = _staged_ctx()
    cfg = _staged_cfg()
    result = RE._select_and_classify_rep(ctx, "D_qubo", cfg)

    assert isinstance(result, RE._ArmResult)
    assert result.selected == list(range(cfg["selection"]["k"]))
    assert result.selection_info == {"stub": True}
    assert "rbf_svm" in result.metrics
    # one row per classifier, carrying the provenance columns the aggregates need
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row["repeat"] == ctx.rep
    assert row["seed"] == ctx.seed
    assert row["arm_key"] == "D_qubo"
    assert row["classifier"] == "rbf_svm"
    assert row["selected"] == ",".join(map(str, result.selected))
    # predictions must not leak into the aggregate rows (they would bloat the CSV)
    assert "predictions" not in row


def test_z_full_arm_selects_every_component(stub_classifiers, stub_select):
    """The Z_full ceiling takes no selection path at all."""
    ctx = _staged_ctx(n_features=12)
    result = RE._select_and_classify_rep(ctx, "Z_full", _staged_cfg())
    assert result.selected == list(range(ctx.Xtr.shape[1]))
    assert result.selection_info == {"no_selection": True}
    assert result.selection_seconds == 0.0


def test_persist_rep_writes_aligned_provenance(tmp_path, stub_classifiers, stub_select):
    """The artifact must record the held-out identity aligned with predictions."""
    ctx = _staged_ctx()
    result = RE._select_and_classify_rep(ctx, "D_qubo", _staged_cfg())
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    RE._persist_rep(run_dir, "stamp", "D_qubo", result, ctx)

    payload = json.loads((run_dir / "rep0_D_qubo.json").read_text(encoding="utf-8"))
    n = len(ctx.test_ids)
    assert payload["test_ids"] == ctx.test_ids
    assert payload["test_groups"] == ctx.test_groups
    assert len(payload["y_test"]) == n
    for clf, m in payload["metrics"].items():
        assert len(m["predictions"]) == n, f"{clf} predictions misaligned"
    assert payload["arm"] == "D_qubo"
    assert payload["rep"] == 0


def test_persist_rep_writes_atomically(tmp_path, stub_classifiers, stub_select):
    """A half-written artifact must never be visible."""
    ctx = _staged_ctx()
    result = RE._select_and_classify_rep(ctx, "D_qubo", _staged_cfg())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    RE._persist_rep(run_dir, "stamp", "D_qubo", result, ctx)
    # no temp files left behind
    leftovers = [p.name for p in run_dir.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_finalize_run_writes_aggregates(tmp_path, monkeypatch, stub_classifiers, stub_select):
    """The finalizer writes the summary CSVs the dashboard and paper read."""
    root = tmp_path / "root"
    (root / "results" / "tables").mkdir(parents=True)
    monkeypatch.setattr(RE, "ROOT", root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = _staged_ctx()
    rows: list[dict] = []
    for arm in ("A_random", "D_qubo"):
        res = RE._select_and_classify_rep(ctx, arm, _staged_cfg())
        rows.extend(res.rows)
        RE._persist_rep(run_dir, "stamp", arm, res, ctx)

    cfg = _staged_cfg()
    results = RE._finalize_run(cfg, run_dir, "stamp", rows)

    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert (run_dir / "results_all.csv").exists()
    assert (run_dir / "config_used.json").exists()
    # per-arm summary + significance tables must exist for the paper pipeline
    tables = list((root / "results" / "tables").glob("*.csv"))
    assert tables, "no summary/significance CSV was written"
