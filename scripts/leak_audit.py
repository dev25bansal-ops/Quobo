"""Audit a run's artifacts for leakage — the paper's claim, checked independently.

The paper asserts the protocol is leak-free: no source photo's crops straddle
the train/test split, and every arm of a repeat is evaluated on the same
held-out set. This command re-derives that evidence from the artifacts a run
left on disk, so a reviewer can check a published run without trusting the
code that produced it.

It never retrains, never replays a split, and never writes into the run
directory. A run without a `run_manifest.json` marked `complete` is refused —
an incomplete run is not evidence.

Usage:
    .venv/Scripts/python.exe -m scripts.leak_audit
    .venv/Scripts/python.exe -m scripts.leak_audit --run-id 20260920_161627
    .venv/Scripts/python.exe -m scripts.leak_audit --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quobo.leak_audit import (  # noqa: E402
    LeakAuditError,
    audit_run,
    format_report,
)


def find_latest_run() -> Path:
    """The run with the most repeats, tie-broken on finish time.

    Same ranking as the dashboard's ``find_complete_run``: directory-name
    order is not a reliable proxy (a diagnostic run with a non-timestamp name
    sorts after every real run).
    """
    experiments = ROOT / "experiments"
    if not experiments.is_dir():
        raise LeakAuditError("no experiments/ directory — run the experiment first")
    best: tuple[int, str, Path] | None = None
    for d in sorted(experiments.iterdir()):
        manifest_path = d / "run_manifest.json"
        if not d.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") != "complete":
            continue
        key = (
            int(manifest.get("reps_completed") or 0),
            str(manifest.get("finished_at") or manifest.get("started_at") or ""),
            d,
        )
        if best is None or key[:2] > best[:2]:
            best = key
    if best is None:
        raise LeakAuditError("no completed run with a run manifest found")
    return best[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="explicit run directory name")
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = parser.parse_args(argv)

    try:
        if args.run_id:
            from pathlib import PurePosixPath, PureWindowsPath

            if (
                PurePosixPath(args.run_id).is_absolute()
                or PureWindowsPath(args.run_id).is_absolute()
                or ".." in PurePosixPath(args.run_id).parts
                or ".." in PureWindowsPath(args.run_id).parts
            ):
                raise LeakAuditError(f"run_id {args.run_id!r} must be a relative name without '..'")
            run_dir = ROOT / "experiments" / args.run_id
            if not run_dir.is_dir():
                raise LeakAuditError(f"experiment run {args.run_id!r} not found")
        else:
            run_dir = find_latest_run()
        audit = audit_run(run_dir)
    except LeakAuditError as exc:
        print(f"leak audit failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(audit.as_dict(), indent=2))
    else:
        print(format_report(audit))
    return 0 if audit.ok else 1


if __name__ == "__main__":
    sys.exit(main())
