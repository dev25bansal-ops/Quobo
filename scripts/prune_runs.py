"""L-04: retention policy for the local ``experiments/`` directory.

The experiment runner writes one directory per run (manifest + per-rep
artifacts + CSVs). Locally that accumulates fast — 18 runs / 38 MB — and every
run is a *result*, so the policy here is deliberately conservative:

- **dry run by default**: the script only REPORTS what it would delete and
  deletes nothing. Deletion requires an explicit ``--yes``;
- ``--keep N`` keeps the N most recent runs (ordered by the manifest's
  ``finished_at``/``started_at`` timestamp, ties broken by ``reps_completed``
  — the same convention the dashboard uses to rank runs);
- ``--max-age DAYS`` keeps every run younger than DAYS days;
- a run whose manifest says ``status == "running"`` is NEVER deleted: a run in
  flight is not garbage, and deleting it would destroy the artifacts of a
  process that is still writing them;
- nothing outside the experiments root can be deleted (path-traversal guard).

Usage:
    .venv/Scripts/python.exe -m scripts.prune_runs --keep 5
    .venv/Scripts/python.exe -m scripts.prune_runs --keep 5 --max-age 30 --yes
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.config import ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("prune")

MANIFEST_NAME = "run_manifest.json"


class PruneError(Exception):
    """Refused operation (path traversal, unknown run, ...)."""


@dataclass
class RunInfo:
    """One directory under ``experiments/`` and what its manifest says."""

    path: Path
    name: str
    status: str
    timestamp: str
    timestamp_dt: datetime | None
    reps_completed: int
    size_bytes: int
    manifest_found: bool = True

    @property
    def age_days(self) -> float | None:
        if self.timestamp_dt is None:
            return None
        return (datetime.now(timezone.utc) - self.timestamp_dt).total_seconds() / 86400.0


@dataclass
class Decision:
    """What the policy decided for one run, and why."""

    run: RunInfo
    delete: bool
    reason: str


@dataclass
class Plan:
    """The full outcome of a policy evaluation (before any deletion)."""

    decisions: list[Decision] = field(default_factory=list)

    @property
    def to_delete(self) -> list[Decision]:
        return [d for d in self.decisions if d.delete]

    @property
    def skipped(self) -> list[Decision]:
        return [d for d in self.decisions if not d.delete]


def _parse_timestamp(value: object) -> tuple[str, datetime | None]:
    """Return the raw timestamp string and its parsed value (None if unparsable).

    Manifests write ISO-8601 with a UTC offset. A legacy manifest may miss the
    field entirely; the caller then falls back to the directory mtime.
    """
    raw = str(value or "").strip()
    if not raw:
        return "", None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw, None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return raw, parsed


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def collect_runs(root: Path) -> list[RunInfo]:
    """Read every run directory under ``root`` and summarize its manifest."""
    runs: list[RunInfo] = []
    if not root.is_dir():
        return runs
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.is_symlink():
            # Symlinked entries are skipped: their real target may live outside
            # the experiments root, and deleting through a link is exactly the
            # accident the traversal guard exists to prevent.
            continue
        manifest_path = d / MANIFEST_NAME
        status, raw_ts, parsed_ts, reps = "no_manifest", "", None, 0
        found = False
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            found = True
            status = str(manifest.get("status") or "unknown")
            raw_ts, parsed_ts = _parse_timestamp(
                manifest.get("finished_at") or manifest.get("started_at")
            )
            reps_raw = manifest.get("reps_completed")
            if isinstance(reps_raw, int):
                reps = reps_raw
        if parsed_ts is None:
            # No usable manifest timestamp: fall back to the directory mtime so
            # age-based pruning still has something to work with.
            try:
                mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)
            except OSError:
                mtime = None
            if mtime is not None:
                raw_ts = raw_ts or mtime.isoformat(timespec="seconds")
                parsed_ts = mtime
        runs.append(
            RunInfo(
                path=d,
                name=d.name,
                status=status,
                timestamp=raw_ts,
                timestamp_dt=parsed_ts,
                reps_completed=reps,
                size_bytes=_dir_size(d),
                manifest_found=found,
            )
        )
    return runs


def rank_runs(runs: list[RunInfo]) -> list[RunInfo]:
    """Order runs newest-first: timestamp, ties broken by ``reps_completed``.

    The tie-break mirrors the dashboard's run ranking, so "the N most recent"
    means the same thing here as it does in the UI.
    """
    return sorted(
        runs,
        key=lambda r: (
            r.timestamp_dt or datetime.min.replace(tzinfo=timezone.utc),
            r.reps_completed,
            r.name,
        ),
        reverse=True,
    )


def build_plan(
    runs: list[RunInfo],
    keep: int | None = None,
    max_age_days: float | None = None,
) -> Plan:
    """Evaluate the retention policy over ``runs`` (no filesystem mutation).

    A run survives when it is among the ``keep`` most recent OR younger than
    ``max_age_days``; it is deleted only when BOTH protections fail. A run in
    the ``running`` state is always kept.
    """
    ranked = rank_runs(runs)
    protected_by_rank = {r.path.name for r in ranked[:keep]} if keep is not None else set()
    plan = Plan()
    for run in ranked:
        if run.status == "running":
            plan.decisions.append(
                Decision(
                    run, delete=False, reason="status is 'running' (run in flight) — never deleted"
                )
            )
            continue
        if run.path.name in protected_by_rank:
            plan.decisions.append(
                Decision(run, delete=False, reason=f"among the {keep} most recent runs (--keep)")
            )
            continue
        age = run.age_days
        if max_age_days is not None and age is not None and age <= max_age_days:
            plan.decisions.append(
                Decision(
                    run,
                    delete=False,
                    reason=f"younger than {max_age_days:g} days (age {age:.1f}d, --max-age)",
                )
            )
            continue
        if max_age_days is not None and age is None:
            plan.decisions.append(
                Decision(run, delete=False, reason="no usable timestamp — kept (--max-age)")
            )
            continue
        reason = "older than the retention window" if max_age_days is not None else "outside --keep"
        plan.decisions.append(Decision(run, delete=True, reason=reason))
    return plan


def _assert_inside_root(root: Path, target: Path) -> None:
    """Refuse any target that does not resolve inside ``root``.

    The policy only ever considers direct children of the experiments root, so
    this is the last line of defence against a crafted name (e.g. a run
    directory that is a symlink, or a relative path with ``..``) escaping the
    experiments tree.
    """
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if target_resolved == root_resolved:
        raise PruneError(f"refusing to delete the experiments root itself: {target}")
    if not target_resolved.is_relative_to(root_resolved):
        raise PruneError(
            f"refusing to delete {target}: it resolves outside the experiments root {root_resolved}"
        )


def delete_run(root: Path, run_path: Path) -> None:
    """Delete one run directory after the containment check."""
    _assert_inside_root(root, run_path)
    shutil.rmtree(run_path)


def _format_size(n: int) -> str:
    return f"{n / (1024 * 1024):.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.1f} kB"


def _print_summary(plan: Plan, root: Path, dry_run: bool) -> None:
    to_delete = plan.to_delete
    total = sum(d.run.size_bytes for d in to_delete)
    print(f"experiments root: {root}")
    print(f"runs inspected  : {len(plan.decisions)}")
    print(f"would delete    : {len(to_delete)} ({_format_size(total)})")
    print(f"kept            : {len(plan.skipped)}")
    if dry_run:
        print("\nDRY RUN — nothing was deleted. Re-run with --yes to apply.")
    print("\n-- to delete --")
    if not to_delete:
        print("  (none)")
    for d in to_delete:
        print(f"  {d.run.name}  [{d.run.status}]  {_format_size(d.run.size_bytes)}  — {d.reason}")
    print("\n-- kept --")
    for d in plan.skipped:
        print(f"  {d.run.name}  [{d.run.status}]  — {d.reason}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retention policy for experiments/ (dry run unless --yes)."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="experiments root to prune (default: <repo>/experiments)",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="keep the N most recent runs",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=None,
        dest="max_age",
        help="keep runs younger than DAYS days",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="actually delete the runs selected by the policy (default: dry run)",
    )
    parser.add_argument(
        "--taxonomy-check",
        action="store_true",
        dest="taxonomy_check",
        help="run the L-10 taxonomy drift check and exit (no pruning)",
    )
    args = parser.parse_args(argv)

    if args.taxonomy_check:
        from src.quobo.data_prep import check_taxonomy_drift

        report = check_taxonomy_drift()
        print(json.dumps(report, indent=2))
        if not report.get("annotations_available"):
            print(f"taxonomy check skipped: {report.get('message', '')}")
            return 0
        return 0 if report.get("ok") else 1

    if args.keep is None and args.max_age is None:
        parser.error(
            "specify --keep N and/or --max-age DAYS; an empty policy would delete every run"
        )
    if args.keep is not None and args.keep < 0:
        parser.error("--keep must be >= 0")

    root = Path(args.root) if args.root else ROOT / "experiments"
    root = root.resolve()
    runs = collect_runs(root)
    if not runs:
        print(f"no runs found under {root}")
        return 0
    plan = build_plan(runs, keep=args.keep, max_age_days=args.max_age)
    _print_summary(plan, root, dry_run=not args.yes)

    if not args.yes:
        return 0

    deleted, failed = [], []
    for decision in plan.to_delete:
        try:
            delete_run(root, decision.run.path)
        except (PruneError, OSError) as exc:
            log.warning("could not delete %s: %s", decision.run.name, exc)
            failed.append((decision.run.name, str(exc)))
            continue
        deleted.append(decision.run.name)
    print(f"\ndeleted {len(deleted)} run(s)")
    if failed:
        print(f"failed to delete {len(failed)} run(s):")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
