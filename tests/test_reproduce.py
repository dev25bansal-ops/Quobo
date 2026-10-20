"""Unit tests for the reproduction entry point (scripts/reproduce_paper.py).

Scope: the PURE parts of the script — the environment-drift detector, the
golden-summary comparison, and the CLI surface. None of these tests runs the
experiment: reproducing the paper's numbers takes minutes and is the
script's job at run time, not the test suite's.

Import-cost note: this module imports ``scripts.reproduce_paper`` at module
scope, and that module deliberately imports NOTHING heavy at module scope
(no ``src.quobo.run_experiment``, no TensorFlow, no pandas). The expensive
imports live inside the functions that need them. A regression that moved
one of them back to module scope would silently add minutes to every CI run,
so ``test_module_scope_is_cheap`` guards it.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import reproduce_paper as rp

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Environment drift detection
# ---------------------------------------------------------------------------


def test_parse_pinned_versions_skips_non_pin_lines():
    """A constraints file is a requirements file with extra syntax.

    The editable install, comments, blank lines and unpinned names are not
    version pins; reporting them as drift would be a false positive.
    """
    text = "\n".join(
        [
            "# pinned environment",
            "-e d:\\quobo",
            "numpy==2.3.1",
            "scipy==1.15.3",
            "pytest",
            "ruff==0.16.5",
            "",
        ]
    )
    pins = rp.parse_pinned_versions(text)
    assert pins == {"numpy": "2.3.1", "scipy": "1.15.3", "ruff": "0.16.5"}
    assert "pytest" not in pins


def test_detect_version_drift_reports_mismatch():
    pins = {"numpy": "2.3.1", "scipy": "1.15.3"}
    installed = {"numpy": "2.3.1", "scipy": "1.14.0"}
    drift = rp.detect_version_drift(pins, installed)
    assert len(drift) == 1
    assert drift[0].package == "scipy"
    assert drift[0].kind == "mismatch"
    assert drift[0].expected == "1.15.3"
    assert drift[0].actual == "1.14.0"


def test_detect_version_drift_reports_missing_package():
    pins = {"numpy": "2.3.1"}
    installed = {"numpy": None}
    drift = rp.detect_version_drift(pins, installed)
    assert len(drift) == 1
    assert drift[0].kind == "not-installed"
    assert drift[0].actual is None


def test_detect_version_drift_is_empty_when_everything_matches():
    pins = {"numpy": "2.3.1"}
    assert rp.detect_version_drift(pins, {"numpy": "2.3.1"}) == []


def test_check_environment_warns_but_does_not_fail_on_drift(monkeypatch):
    """Drift is REPORTED, not fatal, unless --strict.

    A developer machine is not a reproduction environment; a hard failure
    would make the tool useless for the person trying to fix the drift.
    """
    monkeypatch.setattr(rp, "installed_versions", lambda names: {n: "0.0.1" for n in names})
    check = rp.check_environment(strict=False)
    assert check.status == rp.WARN
    assert "drift" in check.detail.lower()
    assert any("DRIFT" in line for line in check.lines)


def test_check_environment_fails_under_strict(monkeypatch):
    monkeypatch.setattr(rp, "installed_versions", lambda names: {n: "0.0.1" for n in names})
    check = rp.check_environment(strict=True)
    assert check.status == rp.FAIL


def test_check_environment_passes_when_nothing_drifts(monkeypatch):
    pins = {"numpy": "2.3.1"}
    monkeypatch.setattr(rp, "parse_pinned_versions", lambda text: pins)
    monkeypatch.setattr(rp, "installed_versions", lambda names: {"numpy": "2.3.1"})
    check = rp.check_environment(strict=True)
    assert check.status == rp.PASS


# ---------------------------------------------------------------------------
# Golden-summary comparison
# ---------------------------------------------------------------------------


def test_tolerance_for_uses_the_sigma_band():
    """The tolerance is sigma * the baseline's own std (Layer-4 convention)."""
    assert rp.tolerance_for(0.02, 1.5) == pytest.approx(0.03)
    # A baseline with no spread gets a 0.01 floor so a tiny change is caught.
    assert rp.tolerance_for(0.0, 1.5) == pytest.approx(0.015)
    assert rp.tolerance_for(None, 1.5) == 0.0


def test_compare_summaries_matching_summary_is_all_ok():
    """A summary that equals the baseline compares equal — every cell OK."""
    baseline = {
        ("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.0325},
        ("QUBO-8", "qsvm"): {"accuracy_mean": 0.3734, "accuracy_std": 0.0311},
    }
    comparisons = rp.compare_summaries(baseline, baseline)
    assert comparisons
    assert all(c.ok for c in comparisons if c.ok is not None)
    assert not [c for c in comparisons if c.status == "MISMATCH"]


def test_compare_summaries_reports_mismatch_with_arm_and_delta():
    """A mutated summary is reported as a MISMATCH naming the arm and the delta.

    This is the whole point of the paper's guard: a number that drifted out of
    the band must be visible, with the offending arm and the size of the
    deviation — not swallowed.
    """
    baseline = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.01}}
    produced = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.5000, "accuracy_std": 0.01}}
    comparisons = rp.compare_summaries(produced, baseline)
    mismatches = [c for c in comparisons if c.status == "MISMATCH"]
    assert len(mismatches) == 1
    bad = mismatches[0]
    assert bad.arm == "QUBO-8"
    assert bad.classifier == "rbf_svm"
    assert bad.metric == "accuracy"
    assert bad.delta == pytest.approx(0.5000 - 0.4082)
    assert bad.ok is False
    assert "QUBO-8" in bad.key


def test_compare_summaries_marks_baseline_cell_missing_from_the_produced_run():
    """The baseline has a cell the produced run does not -> MISSING (a failure).

    A cell the baseline knows about and the reproduction does not produce is
    a reproducibility failure, not a curiosity: something the paper reported
    is simply absent.
    """
    baseline = {
        ("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.03},
        ("LASSO-8", "qsvm"): {"accuracy_mean": 0.3846, "accuracy_std": 0.02},
    }
    produced = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.03}}
    comparisons = rp.compare_summaries(produced, baseline)
    missing = [c for c in comparisons if c.status == "MISSING"]
    assert len(missing) == 1
    assert missing[0].arm == "LASSO-8"
    assert missing[0].ok is False


def test_compare_summaries_marks_metric_without_a_baseline():
    """The accuracy-only fallback baseline: macro_f1 has nothing to compare to.

    Such a cell is NOT comparable, so it is not counted as a pass and not
    counted as a failure either.
    """
    baseline = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.03}}
    produced = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.03}}
    comparisons = rp.compare_summaries(produced, baseline)
    no_baseline = [c for c in comparisons if c.status == "NO_BASELINE"]
    assert len(no_baseline) == 1
    assert no_baseline[0].metric == "macro_f1"
    assert no_baseline[0].ok is None


def test_verdict_reports_failure_with_the_first_offending_cell():
    baseline = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.001}}
    produced = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.5500, "accuracy_std": 0.001}}
    verdict = rp.render_verdict(rp.compare_summaries(produced, baseline))
    assert verdict.startswith("VERDICT: FAIL")
    assert "QUBO-8" in verdict


def test_verdict_reports_pass_when_everything_is_within_the_band():
    baseline = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4082, "accuracy_std": 0.03}}
    produced = {("QUBO-8", "rbf_svm"): {"accuracy_mean": 0.4100, "accuracy_std": 0.03}}
    verdict = rp.render_verdict(rp.compare_summaries(produced, baseline))
    assert verdict.startswith("VERDICT: PASS")


# ---------------------------------------------------------------------------
# Summary CSV parsing
# ---------------------------------------------------------------------------


def _write_summary_csv(path: Path) -> None:
    """Write a summary CSV in the shape run_experiment produces (3 header rows)."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["", "", "accuracy", "accuracy", "macro_f1", "macro_f1"])
        writer.writerow(["", "", "mean", "std", "mean", "std"])
        writer.writerow(["arm", "classifier", "", "", "", ""])
        writer.writerow(["QUBO-8", "rbf_svm", "0.4082", "0.0325", "0.3893", "0.0293"])
        writer.writerow(["LASSO-8", "qsvm", "0.3846", "0.0238", "0.3525", "0.0273"])


def test_read_summary_csv_parses_the_three_row_header(tmp_path):
    path = tmp_path / "summary_x.csv"
    _write_summary_csv(path)
    table = rp.read_summary_csv(path)
    assert set(table) == {("QUBO-8", "rbf_svm"), ("LASSO-8", "qsvm")}
    row = table[("QUBO-8", "rbf_svm")]
    assert row["accuracy_mean"] == pytest.approx(0.4082)
    assert row["accuracy_std"] == pytest.approx(0.0325)
    assert row["macro_f1_mean"] == pytest.approx(0.3893)


def test_read_summary_csv_rejects_a_file_that_is_not_a_summary(tmp_path):
    path = tmp_path / "nope.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary CSV"):
        rp.read_summary_csv(path)


def test_summarise_results_frame_matches_the_run_aggregation(tmp_path):
    """The produced summary must aggregate exactly like run_experiment does."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "arm": ["QUBO-8"] * 4,
            "classifier": ["rbf_svm"] * 2 + ["qsvm"] * 2,
            "accuracy": [0.40, 0.42, 0.37, 0.38],
            "macro_f1": [0.38, 0.40, 0.33, 0.34],
        }
    )
    table = rp.summarise_results_frame(frame)
    assert table[("QUBO-8", "rbf_svm")]["accuracy_mean"] == pytest.approx(0.41)
    assert table[("QUBO-8", "rbf_svm")]["accuracy_std"] == pytest.approx(0.0141, abs=1e-4)
    assert table[("QUBO-8", "qsvm")]["macro_f1_mean"] == pytest.approx(0.335)


# ---------------------------------------------------------------------------
# Baseline resolution
# ---------------------------------------------------------------------------


def test_resolve_baseline_prefers_an_explicit_summary(tmp_path):
    path = tmp_path / "custom_summary.csv"
    _write_summary_csv(path)
    baseline = rp.resolve_baseline(golden_summary=path)
    assert baseline.table
    assert baseline.sigma == rp.DEFAULT_SIGMA_TOLERANCE


def test_resolve_baseline_falls_back_to_the_harness_constants(monkeypatch, tmp_path):
    """A fresh clone has no results/ (it is gitignored) — the harness is the fallback."""
    # Point the results/ lookup at an empty tree so the on-disk summary cannot
    # be found; the harness constants must then carry the comparison.
    monkeypatch.setattr(rp, "ROOT", tmp_path)
    baseline = rp.resolve_baseline()
    assert baseline.source.startswith("tests/test_golden_results.py")
    # The harness baseline is accuracy-only; macro_f1 has nothing to compare to.
    assert all("accuracy_mean" in stats for stats in baseline.table.values())


def test_harness_tolerance_is_shared_with_the_script():
    """The tolerance multiplier must come from the Layer-4 harness, not a copy.

    Copying the number would let the script and the paper's guard drift apart
    — the exact failure this module exists to prevent.
    """
    harness = rp._load_harness_module()
    if harness is None:  # pragma: no cover - only when pytest is unavailable
        pytest.skip("the Layer-4 harness is not importable here")
    assert float(harness.SIGMA_TOLERANCE) == rp.DEFAULT_SIGMA_TOLERANCE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero():
    """`--help` must work without touching the data or running anything."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reproduce_paper", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout


def test_dry_run_exits_zero_on_the_current_tree():
    """The dry run performs every check EXCEPT the experiment, and must pass.

    This is the cheap CI entry point: if it fails, the environment or the
    dataset is not in a state where the reproduction could even start.
    """
    result = subprocess.run(
        [sys.executable, "-m", "scripts.reproduce_paper", "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: PASS" in result.stdout
    # Every check must be reported, not silently skipped.
    for name in ("config", "environment", "data"):
        assert f"{name}:" in result.stdout


def test_module_scope_stays_cheap():
    """The module must not import the pipeline at module scope.

    ``src.quobo.run_experiment`` used to cost ~208s to import (TensorFlow at
    module scope). A regression that moves a heavy import back up here would
    add minutes to every test run, so the budget is asserted, not assumed.
    """
    code = (
        "import time, sys; t=time.perf_counter(); "
        "import scripts.reproduce_paper; "
        "print(time.perf_counter()-t); "
        "print('tensorflow' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    elapsed = float(result.stdout.splitlines()[0])
    assert elapsed < 10.0, (
        f"module import took {elapsed:.1f}s — a heavy import leaked to module scope"
    )
    assert result.stdout.splitlines()[1].strip() == "False"


# ---------------------------------------------------------------------------
# The README's test count
# ---------------------------------------------------------------------------

# The marker the README carries so the number is machine-readable instead of
# being buried in prose where nobody can check it.
_COUNT_MARKER = re.compile(r"<!--\s*test-count:\s*(\d+)\s*-->")

# How far the README's figure may drift from the real collection before the
# guard fails. A hard equality check would be a landmine: the suite grows
# every time a test is added, and the check would then fail for a reason
# that has nothing to do with the code under test. A tolerance keeps the
# guard meaningful (a stale count by dozens IS a documentation bug) without
# making it a coin flip.
_COUNT_TOLERANCE = 0.25


def _readme_stated_count() -> int | None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = _COUNT_MARKER.search(text)
    return int(match.group(1)) if match else None


def test_readme_carries_a_machine_readable_test_count():
    """The README must carry the count in a form a test can check.

    A count written only in prose cannot be verified, so it silently rots.
    """
    stated = _readme_stated_count()
    assert stated is not None, (
        "README.md has no '<!-- test-count: N -->' marker; the test count must be "
        "machine-readable so it can be checked against a real collection"
    )
    assert stated > 0


@pytest.mark.skipif(
    os.environ.get("QUOBO_CHECK_TEST_COUNT") != "1",
    reason="set QUOBO_CHECK_TEST_COUNT=1 to verify the README's test count "
    "(it re-collects the whole suite, so it is opt-in)",
)
def test_readme_test_count_is_not_stale():
    """The README's stated count must still be close to a real collection.

    WHY opt-in: verifying the number means re-collecting the whole suite
    (~15-40s), which is too expensive to do on every run of every test file.
    It is checked in CI, where the count actually matters.

    The tolerance is deliberate. The suite grows every time a test lands, so
    an exact-equality check would fail for reasons unrelated to the code —
    and a check that cries wolf gets ignored. A stale count by dozens of
    tests is still caught.
    """
    stated = _readme_stated_count()
    assert stated is not None, "README.md has no '<!-- test-count: N -->' marker"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    match = re.search(r"(\d+) tests? collected", result.stdout)
    assert match, f"could not parse the collected count from:\n{result.stdout[-2000:]}"
    actual = int(match.group(1))

    drift = abs(actual - stated)
    allowed = max(5, int(stated * _COUNT_TOLERANCE))
    assert drift <= allowed, (
        f"README.md states {stated} tests but the suite collects {actual} "
        f"(drift {drift} > {allowed}). Update the '<!-- test-count: N -->' marker "
        f"in README.md, or run: pytest --collect-only -q | tail -1"
    )
