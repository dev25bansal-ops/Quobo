"""One command from a fresh clone to the paper's numbers.

Why this exists (catalog Part 4, "scripts/reproduce_paper.py"): the paper's
credibility rests on reproducibility, and today the only way to get there is to
know the right incantations (the config, the pinned environment, the dataset
layout, the golden baseline) and to read the runbook carefully. A reviewer
should not need the runbook. This script is the runbook, executable.

The check sequence mirrors what a reviewer must be able to trust, in order:

1. **Config** — the frozen paper config is loaded and validated through the
   real loader/validator (``src.quobo.config.load_config`` /
   ``validate_config``), so a malformed config fails here and not 20 minutes
   into a run.
2. **Environment** — the interpreter version and the installed package
   versions are compared with ``constraints.txt`` (the exact pins that
   produced the paper numbers). Drift is REPORTED, not fatal: a developer
   machine is not a reproduction environment, and a hard failure would make
   the tool useless for the person trying to fix the drift. ``--strict``
   turns drift into a failure.
3. **Data** — the on-disk dataset is checked with
   ``src.quobo.data_validation.validate_data`` (the offline contract
   validator). A dataset that has not been downloaded yet is a SKIP with an
   actionable message, not a failure.
4. **Run** — the experiment runs, or an existing completed run is reused
   (``--run-id`` / ``--skip-run``). The expensive part is the only part that
   cannot be dry-run.
5. **Golden comparison** — the produced summary is compared against the
   frozen paper baseline under the SAME tolerance convention the Layer-4
   statistical harness uses (``tests/test_golden_results.py``): a metric must
   sit within +/-1.5 sigma of the baseline's own standard deviation
   (0.01 when the baseline std is 0). A metric that drifts out of that band
   is a FAILURE, not a warning — that is the whole point of the paper's
   guard.

Usage:
    .venv/Scripts/python.exe -m scripts.reproduce_paper --dry-run
    .venv/Scripts/python.exe -m scripts.reproduce_paper
    .venv/Scripts/python.exe -m scripts.reproduce_paper --run-id 20260920_161627
    .venv/Scripts/python.exe -m scripts.reproduce_paper --skip-run --strict

Exit codes: 0 = every check passed, 1 = a check failed (mismatch, invalid
config, or drift under --strict), 2 = the script could not run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The Layer-4 harness is a FIXED location relative to this file, not relative
# to ROOT: ROOT is the repo root that a caller may point elsewhere (a
# relocated checkout, a test fixture), but the harness that defines the
# paper's baseline and its tolerance always lives next to this script.
HARNESS_PATH = Path(__file__).resolve().parents[1] / "tests" / "test_golden_results.py"

log = logging.getLogger("reproduce")

# The paper's config. Frozen: this is the run the paper's Table 1 reports.
DEFAULT_CONFIG = "configs/experiment_6class.yaml"

# The golden baseline lives in the Layer-4 harness (tests/test_golden_results.py)
# so there is exactly ONE definition of the paper's numbers and of the
# tolerance convention. These are the fallbacks for the case where the test
# module cannot be imported (a production install without pytest).
DEFAULT_GOLDEN_RUN = "20260920_161627"
DEFAULT_SIGMA_TOLERANCE = 1.5

# Python the paper was produced on (README + constraints.txt).
REQUIRED_PYTHON = (3, 12)

# Statuses used across the report.
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

METRICS = ("accuracy", "macro_f1")


# --------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_reproduce.py)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionDrift:
    """One package whose installed version differs from the pinned one."""

    package: str
    expected: str | None
    actual: str | None
    kind: str  # "mismatch" | "missing" | "not-installed"


@dataclass(frozen=True)
class MetricComparison:
    """One (arm, classifier, metric) cell of the golden comparison."""

    arm: str
    classifier: str
    metric: str
    baseline: float | None
    produced: float | None
    delta: float | None
    tolerance: float | None
    status: str  # "OK" | "MISMATCH" | "MISSING" | "NO_BASELINE"
    ok: bool | None  # None = not comparable (no baseline)

    @property
    def key(self) -> str:
        return f"{self.arm}/{self.classifier}"


def parse_pinned_versions(text: str) -> dict[str, str]:
    """Extract the ``name==version`` pins from a constraints file.

    The file also carries an editable install (``-e d:\\quobo``) and a few
    unpinned names (``pytest``); those are not version pins and are skipped
    rather than being reported as drift.

    WHY: a constraints file is a *requirements* file with extra syntax, so
    parsing it as "every non-empty line is a pin" produces false drift.
    """
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.strip()
        version = version.strip()
        if name and version:
            pins[name] = version
    return pins


def installed_versions(names: list[str]) -> dict[str, str | None]:
    """Resolve the installed version of each name (None when not installed)."""
    found: dict[str, str | None] = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def detect_version_drift(
    pins: dict[str, str], installed: dict[str, str | None]
) -> list[VersionDrift]:
    """Compare pinned versions with what is installed.

    Three failure shapes, all reported:
    - ``mismatch``: installed, but a different version;
    - ``not-installed``: pinned, but absent from the environment;
    - ``missing``: the pin could not be resolved at all (defensive; the
      installed map is expected to carry an entry for every pin).
    """
    drift: list[VersionDrift] = []
    for name, expected in sorted(pins.items()):
        actual = installed.get(name)
        if actual is None:
            kind = "not-installed" if name in installed else "missing"
            drift.append(VersionDrift(package=name, expected=expected, actual=None, kind=kind))
        elif actual != expected:
            drift.append(
                VersionDrift(package=name, expected=expected, actual=actual, kind="mismatch")
            )
    return drift


def tolerance_for(std: float | None, sigma: float) -> float:
    """The Layer-4 tolerance band: +/- sigma * the baseline's own std.

    Mirrors ``tests/test_golden_results.py``: a metric must sit within
    ``SIGMA_TOLERANCE`` standard deviations of the frozen baseline, and a
    baseline with no spread (std 0) gets a 0.01 floor so a tiny change is
    still caught. Same convention, one number, two places.
    """
    if std is None:
        return 0.0
    return sigma * (std if std > 0 else 0.01)


def compare_summaries(
    produced: dict[tuple[str, str], dict[str, float]],
    baseline: dict[tuple[str, str], dict[str, float]],
    sigma: float = DEFAULT_SIGMA_TOLERANCE,
) -> list[MetricComparison]:
    """Compare a produced summary against the frozen baseline, cell by cell.

    ``produced``/``baseline`` map ``(arm, classifier)`` to a dict of metric
    statistics (``accuracy_mean``, ``accuracy_std``, ...).

    Statuses:
    - ``OK``: the produced mean is inside the baseline's band;
    - ``MISMATCH``: it is outside — a real regression, reported as a FAIL;
    - ``MISSING``: the baseline has the cell but the produced run does not;
    - ``NO_BASELINE``: the baseline does not carry that metric (e.g. the
      accuracy-only fallback), so the cell is not comparable and is NOT
      counted as a pass.
    """
    comparisons: list[MetricComparison] = []
    for key in sorted(set(produced) | set(baseline)):
        arm, classifier = key
        for metric in METRICS:
            base_mean = baseline.get(key, {}).get(f"{metric}_mean")
            base_std = baseline.get(key, {}).get(f"{metric}_std")
            prod_mean = produced.get(key, {}).get(f"{metric}_mean")
            if base_mean is None:
                comparisons.append(
                    MetricComparison(
                        arm=arm,
                        classifier=classifier,
                        metric=metric,
                        baseline=None,
                        produced=prod_mean,
                        delta=None,
                        tolerance=None,
                        status="NO_BASELINE",
                        ok=None,
                    )
                )
                continue
            if prod_mean is None:
                comparisons.append(
                    MetricComparison(
                        arm=arm,
                        classifier=classifier,
                        metric=metric,
                        baseline=base_mean,
                        produced=None,
                        delta=None,
                        tolerance=None,
                        status="MISSING",
                        ok=False,
                    )
                )
                continue
            tolerance = tolerance_for(base_std, sigma)
            delta = prod_mean - base_mean
            comparisons.append(
                MetricComparison(
                    arm=arm,
                    classifier=classifier,
                    metric=metric,
                    baseline=base_mean,
                    produced=prod_mean,
                    delta=delta,
                    tolerance=tolerance,
                    status="OK" if abs(delta) <= tolerance else "MISMATCH",
                    ok=abs(delta) <= tolerance,
                )
            )
    return comparisons


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_comparison_table(comparisons: list[MetricComparison], metric: str) -> str:
    """One PASS/FAIL table per metric, one row per (arm, classifier)."""
    rows = [c for c in comparisons if c.metric == metric]
    if not rows:
        return f"  (no {metric} cells to compare)"
    header = (
        f"  {'arm / classifier':<26} {'baseline':>9} {'produced':>9} "
        f"{'delta':>9} {'+/- band':>9}  status"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for c in rows:
        lines.append(
            f"  {c.key:<26} {_fmt(c.baseline):>9} {_fmt(c.produced):>9} "
            f"{_fmt(c.delta):>9} {_fmt(c.tolerance):>9}  {c.status}"
        )
    return "\n".join(lines)


def render_verdict(comparisons: list[MetricComparison]) -> str:
    """The final one-line verdict."""
    comparable = [c for c in comparisons if c.ok is not None]
    failed = [c for c in comparable if not c.ok]
    skipped = [c for c in comparisons if c.ok is None]
    if failed:
        first = failed[0]
        return (
            f"VERDICT: FAIL — {len(failed)}/{len(comparable)} metric(s) outside the "
            f"golden band; first: {first.key} {first.metric} "
            f"delta {first.delta:+.4f} vs band +/-{first.tolerance:.4f}"
        )
    if not comparable:
        return "VERDICT: FAIL — nothing could be compared against the baseline"
    suffix = f" ({len(skipped)} cell(s) had no baseline)" if skipped else ""
    return f"VERDICT: PASS — {len(comparable)} metric(s) within the golden band{suffix}"


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def read_summary_csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Read a run summary CSV into ``{(arm, classifier): {stat: value}}``.

    The summary CSVs are written by ``run_experiment`` as a grouped, rounded
    frame, so the file has a three-row header (metric names, statistic names,
    index names) before the data. The header is parsed rather than assumed:
    the column order is read from the file itself, so a future metric cannot
    silently shift the columns.
    """
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    index_row = next(
        (
            i
            for i, row in enumerate(rows)
            if len(row) >= 2 and row[0].strip() == "arm" and row[1].strip() == "classifier"
        ),
        None,
    )
    if index_row is None or index_row < 2:
        raise ValueError(
            f"{path.name} does not look like a run summary CSV (no arm/classifier header)"
        )
    metric_row, stat_row = rows[index_row - 2], rows[index_row - 1]
    columns = [
        (metric_row[i].strip(), stat_row[i].strip())
        for i in range(2, min(len(metric_row), len(stat_row)))
    ]
    table: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows[index_row + 1 :]:
        if len(row) < 2 or not row[0].strip():
            continue
        key = (row[0].strip(), row[1].strip())
        stats: dict[str, float] = {}
        for i, (metric, stat) in enumerate(columns, start=2):
            if i >= len(row) or not row[i].strip():
                continue
            stats[f"{metric}_{stat}"] = float(row[i])
        table[key] = stats
    return table


def summarise_results_frame(frame) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate a run's ``results_all.csv`` into the same shape.

    Uses the same aggregation as ``run_experiment`` (group by arm and
    classifier, mean and std of accuracy and macro_f1) so the produced
    numbers are directly comparable to the golden summary CSV.
    """
    grouped = frame.groupby(["arm", "classifier"])[["accuracy", "macro_f1"]].agg(["mean", "std"])
    grouped = grouped.round(4)
    table: dict[tuple[str, str], dict[str, float]] = {}
    for (arm, classifier), row in grouped.iterrows():
        table[(str(arm), str(classifier))] = {
            f"{metric}_{stat}": float(row[(metric, stat)])
            for metric in ("accuracy", "macro_f1")
            for stat in ("mean", "std")
        }
    return table


def _load_harness_module() -> Any | None:
    """Import the Layer-4 harness so the baseline and tolerance are shared.

    The harness is the paper's guard: it holds the frozen baseline
    (``GOLDEN_ACCURACY``) and the tolerance convention
    (``SIGMA_TOLERANCE``). Importing it — instead of copying the numbers —
    is what keeps the two from drifting apart. It needs pytest, so a
    production install without the dev extra simply falls back to the
    defaults below.
    """
    path = HARNESS_PATH
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_quobo_golden_harness", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — a missing harness must not break the run
        log.debug("golden harness not importable: %s: %s", type(exc).__name__, exc)
        return None
    return module


@dataclass
class Baseline:
    """The frozen paper baseline the produced run is compared against."""

    table: dict[tuple[str, str], dict[str, float]]
    run_id: str
    sigma: float
    source: str


def resolve_baseline(golden_summary: Path | None = None, golden_run: str | None = None) -> Baseline:
    """Find the frozen baseline and the tolerance convention.

    Resolution order:
    1. an explicit ``--golden-summary`` path;
    2. ``results/tables/summary_<run>.csv`` — the paper's own summary
       artifact (present in a working tree; ``results/`` is not tracked, so
       a fresh clone falls through to 3);
    3. the Layer-4 harness constants (accuracy only).

    The tolerance multiplier always comes from the harness when it can be
    imported, so the script and the guard use one number.
    """
    harness = _load_harness_module()
    run_id = golden_run or getattr(harness, "GOLDEN_RUN", DEFAULT_GOLDEN_RUN)
    sigma = float(getattr(harness, "SIGMA_TOLERANCE", DEFAULT_SIGMA_TOLERANCE))

    candidates: list[Path] = []
    if golden_summary is not None:
        candidates.append(golden_summary)
    else:
        candidates.append(ROOT / "results" / "tables" / f"summary_{run_id}.csv")
    for candidate in candidates:
        if candidate.is_file():
            return Baseline(
                table=read_summary_csv(candidate),
                run_id=run_id,
                sigma=sigma,
                source=str(
                    candidate.relative_to(ROOT) if candidate.is_relative_to(ROOT) else candidate
                ),
            )

    # Fall back to the harness's frozen accuracy baseline.
    accuracy = getattr(harness, "GOLDEN_ACCURACY", None)
    if accuracy:
        table = {
            (arm, classifier): {"accuracy_mean": mean, "accuracy_std": std}
            for (arm, classifier), (mean, std) in accuracy.items()
        }
        return Baseline(
            table=table,
            run_id=run_id,
            sigma=sigma,
            source="tests/test_golden_results.py::GOLDEN_ACCURACY (accuracy only)",
        )
    raise FileNotFoundError(
        f"no golden baseline found: neither {candidates[0]} nor the Layer-4 harness "
        "constants are available. Pass --golden-summary <path to a summary CSV>."
    )


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


@dataclass
class Check:
    """The outcome of one check in the reproduction sequence."""

    name: str
    status: str
    detail: str
    lines: list[str]


def check_config(config_path: str) -> Check:
    """Load and validate the frozen config through the real loader."""
    from src.quobo.config import load_config, validate_config

    try:
        cfg = load_config(config_path)
        model = validate_config(cfg)
    except Exception as exc:  # noqa: BLE001 — any loader error is a config failure
        return Check("config", FAIL, f"{config_path}: {type(exc).__name__}: {exc}", [])
    normalized = model.model_dump()
    detail = (
        f"{config_path}: {len(normalized['data']['classes'])} classes, "
        f"k={normalized['selection']['k']}, PCA-{normalized['features']['pca_components']}, "
        f"{normalized['experiment']['n_repeats']} repeats, "
        f"{len(normalized['experiment']['arms'])} arms"
    )
    return Check("config", PASS, detail, [])


def check_environment(strict: bool) -> Check:
    """Compare the interpreter and the installed packages with the pins."""
    lines: list[str] = []
    actual_py = platform.python_version()
    expected_py = ".".join(str(p) for p in REQUIRED_PYTHON)
    py_ok = tuple(sys.version_info[:2]) == REQUIRED_PYTHON
    lines.append(f"python {actual_py} (required {expected_py}) — {'ok' if py_ok else 'DRIFT'}")

    constraints = ROOT / "constraints.txt"
    if not constraints.is_file():
        return Check(
            "environment",
            FAIL if strict else WARN,
            "constraints.txt is missing — cannot verify the pinned versions",
            lines,
        )
    pins = parse_pinned_versions(constraints.read_text(encoding="utf-8"))
    installed = installed_versions(list(pins))
    drift = detect_version_drift(pins, installed)
    lines.append(f"pinned packages checked: {len(pins)}")
    for item in drift:
        if item.kind == "mismatch":
            lines.append(f"  DRIFT {item.package}: installed {item.actual}, pinned {item.expected}")
        elif item.kind == "not-installed":
            lines.append(f"  DRIFT {item.package}: not installed (pinned {item.expected})")
        else:
            lines.append(f"  DRIFT {item.package}: version could not be resolved")

    if drift or not py_ok:
        summary = f"{len(drift)} package(s) drifted from constraints.txt" + (
            "" if py_ok else f"; python {actual_py} != {expected_py}"
        )
        if strict:
            return Check("environment", FAIL, summary, lines)
        return Check(
            "environment",
            WARN,
            summary + " (reported, not fatal without --strict)",
            lines,
        )
    return Check("environment", PASS, f"python {actual_py}; {len(pins)} pins match", lines)


def check_data(mode: str, config_path: str = DEFAULT_CONFIG) -> Check:
    """Check the on-disk dataset.

    ``full`` runs the complete ``validate_data`` contract (every crop is
    decoded, the embedding cache is checked against the crop inventory).
    ``quick`` runs the fingerprint identity check only — the same
    ``features.crops_fingerprint`` the validator mirrors — which is cheap
    enough for CI. ``off`` skips the check.
    """
    if mode == "off":
        return Check("data", SKIP, "skipped (--data-check off)", [])

    from src.quobo.config import load_config

    cfg = load_config(config_path)
    classes = list(cfg["data"]["classes"])
    crops_dir = ROOT / "data" / "processed" / "crops"
    crop_meta = ROOT / "data" / "processed" / "data_prep_stats.json"
    features_dir = ROOT / "data" / "features"
    tag = "raw_" + "_".join(sorted(classes))
    cache = features_dir / f"{tag}.npz"
    cache_meta = features_dir / f"{tag}_meta.json"

    if not crops_dir.is_dir():
        return Check(
            "data",
            SKIP,
            "dataset not present (data/processed/crops is missing) — download it with "
            "`.venv/Scripts/python.exe -m src.quobo.data_prep "
            f"{DEFAULT_CONFIG}` and re-run",
            [],
        )

    if mode == "quick":
        from src.quobo.features import crops_fingerprint

        fingerprint = crops_fingerprint(crops_dir, classes)
        lines = [f"crops: {fingerprint['n_crops']} files, files_hash {fingerprint['files_hash']}"]
        if cache_meta.is_file():
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            cached = meta.get("crops_fingerprint")
            if cached == fingerprint:
                lines.append("embedding cache fingerprint matches the crop inventory")
                return Check("data", PASS, "crop inventory and cache fingerprint agree", lines)
            lines.append("embedding cache fingerprint does NOT match the crop inventory")
            return Check("data", FAIL, "crop inventory and cache fingerprint disagree", lines)
        lines.append("no embedding cache metadata — the run will recompute the embeddings")
        return Check("data", PASS, "crop inventory verified (no cache to cross-check)", lines)

    from src.quobo.data_validation import validate_data

    report = validate_data(
        crops_dir=crops_dir,
        crop_meta=crop_meta if crop_meta.is_file() else None,
        classes=classes,
        cache=cache if cache.is_file() else None,
        cache_meta=cache_meta if cache_meta.is_file() else None,
    )
    lines = [f"crops checked under {crops_dir.relative_to(ROOT)}"]
    for warning in report.get("warnings", []):
        lines.append(f"  note: {warning}")
    if report["ok"]:
        return Check("data", PASS, "data contract validated", lines)
    for error in report.get("errors", []):
        lines.append(f"  ERROR {error['code']}: {error['path']}: {error['message']}")
    return Check("data", FAIL, f"{len(report['errors'])} data contract error(s)", lines)


def find_completed_run(run_id: str | None) -> tuple[Path, dict]:
    """Locate a completed run under ``experiments/``.

    Reuses the dashboard's ``find_complete_run`` so the provenance rules
    (manifest present, ``status == "complete"``, at least one completed
    repeat) are one implementation, not two that can disagree.
    """
    from scripts.pollution_dashboard import find_complete_run

    return find_complete_run(run_id)


def run_experiment(config_path: str, run_id: str) -> Path:
    """Run the paper experiment and return the run directory."""
    from src.quobo.config import load_config
    from src.quobo.run_experiment import run

    cfg = load_config(config_path)
    logging.getLogger().setLevel(logging.INFO)
    run(cfg, run_id=run_id, config_path=config_path)
    return ROOT / "experiments" / run_id


def produced_summary_from_run(run_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Recompute a run's summary from its own per-repeat artifacts.

    The run's ``results_all.csv`` is the authority: it is what the summary
    was derived from in the first place, and recomputing it means the
    comparison consumes the run's artifacts rather than a derived file that
    might be stale.
    """
    import pandas as pd

    results_csv = run_dir / "results_all.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"{results_csv} is missing — the run did not produce results")
    return summarise_results_frame(pd.read_csv(results_csv))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's numbers from a fresh clone.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 = all checks passed, 1 = a check failed, 2 = the script could not run"
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"frozen experiment config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform every check EXCEPT the expensive experiment",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="reuse this completed run under experiments/ instead of running one",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="reuse the most recent completed run instead of running one",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat environment drift as a failure (a dev machine is not a paper machine)",
    )
    parser.add_argument(
        "--data-check",
        choices=("full", "quick", "off"),
        default=None,
        help=(
            "dataset validation depth: full = the complete validate_data contract, "
            "quick = fingerprint identity only (default: full, or quick with --dry-run), "
            "off = skip"
        ),
    )
    parser.add_argument(
        "--golden-summary",
        type=Path,
        default=None,
        help="summary CSV to compare against (default: results/tables/summary_<run>.csv)",
    )
    parser.add_argument(
        "--golden-run",
        default=None,
        help=f"run id of the golden baseline (default: {DEFAULT_GOLDEN_RUN})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    data_mode = args.data_check or ("quick" if args.dry_run else "full")
    checks: list[Check] = []
    failures = 0

    def emit(check: Check) -> None:
        nonlocal failures
        checks.append(check)
        if check.status == FAIL:
            failures += 1
        print(f"[{check.status}] {check.name}: {check.detail}")
        for line in check.lines:
            print(line)

    print("Quobo — paper reproduction")
    print(f"repo: {ROOT}")
    print(f"mode: {'dry-run (no experiment)' if args.dry_run else 'full reproduction'}")
    print("-" * 78)

    # 1. Config — through the real loader/validator.
    config_check = check_config(args.config)
    emit(config_check)
    if config_check.status == FAIL:
        print("-" * 78)
        print("VERDICT: FAIL — the frozen config does not validate; stopping before the run")
        return 1

    # 2. Environment — drift is reported, not fatal (unless --strict).
    emit(check_environment(args.strict))

    # 3. Data — the dataset may legitimately not be downloaded yet.
    emit(check_data(data_mode, args.config))

    # A failed precondition aborts BEFORE the expensive run: running the
    # experiment on an invalid config or a drifted environment would burn
    # twenty minutes to produce numbers nobody should trust.
    if failures:
        print("-" * 78)
        print(
            f"VERDICT: FAIL — {failures} precondition check(s) failed; the experiment was not run"
        )
        return 1

    # 4. Run (or reuse a completed one).
    run_dir: Path | None = None
    if args.dry_run:
        print(f"[{SKIP}] run: skipped (--dry-run)")
    else:
        try:
            if args.run_id:
                run_dir, manifest = find_completed_run(args.run_id)
                emit(
                    Check(
                        "run",
                        PASS,
                        f"reusing completed run {run_dir.name} "
                        f"({manifest.get('reps_completed')} repeats)",
                        [],
                    )
                )
            elif args.skip_run:
                run_dir, manifest = find_completed_run(None)
                emit(
                    Check(
                        "run",
                        PASS,
                        f"reusing most recent completed run {run_dir.name} "
                        f"({manifest.get('reps_completed')} repeats)",
                        [],
                    )
                )
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                print(f"[....] run: executing the experiment (run id {stamp}) — this takes a while")
                run_dir = run_experiment(args.config, stamp)
                emit(Check("run", PASS, f"experiment finished: {run_dir.name}", []))
        except Exception as exc:  # noqa: BLE001 — a run failure is a reproduction failure
            emit(Check("run", FAIL, f"{type(exc).__name__}: {exc}", []))
            print("-" * 78)
            print("VERDICT: FAIL — the experiment could not be produced")
            return 1

    # 5. Golden comparison.
    if run_dir is None:
        print("-" * 78)
        print("VERDICT: PASS — dry run: all cheap checks passed, no experiment was executed")
        return 0

    try:
        baseline = resolve_baseline(args.golden_summary, args.golden_run)
    except Exception as exc:  # noqa: BLE001 — no baseline means no comparison
        emit(Check("golden", FAIL, f"{type(exc).__name__}: {exc}", []))
        print("-" * 78)
        print("VERDICT: FAIL — no golden baseline available to compare against")
        return 1

    produced = produced_summary_from_run(run_dir)
    comparisons = compare_summaries(produced, baseline.table, baseline.sigma)
    print(
        f"[....] golden: baseline {baseline.source} (run {baseline.run_id}, "
        f"band +/-{baseline.sigma} sigma)"
    )
    for metric in METRICS:
        print(f"  {metric}:")
        print(render_comparison_table(comparisons, metric))
    print("-" * 78)
    verdict = render_verdict(comparisons)
    print(verdict)
    failed = [c for c in comparisons if c.ok is False]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
