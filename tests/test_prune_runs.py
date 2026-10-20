"""Unit tests for the experiments/ retention policy (L-04).

The policy's safety properties are what these tests pin:

- the default is a DRY RUN — nothing is deleted without ``--yes``;
- ``--yes`` deletes exactly the runs the policy selected;
- a run whose manifest says ``status == "running"`` is NEVER deleted;
- nothing outside the experiments root can be deleted (path traversal).
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_run(
    root: Path,
    name: str,
    *,
    status: str = "complete",
    finished_at: str | None = None,
    reps: int = 1,
) -> Path:
    """Create a run directory with a manifest that looks like the real one."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    stamp = finished_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "run_id": name,
        "status": status,
        "started_at": stamp,
        "finished_at": None if status == "running" else stamp,
        "reps_completed": reps,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "results_all.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return run_dir


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")


def test_dry_run_deletes_nothing(tmp_path, capsys):
    """Default behaviour is a report, not a mutation."""
    from scripts.prune_runs import main

    root = tmp_path / "experiments"
    _make_run(root, "old_run", finished_at=_iso(30))
    _make_run(root, "new_run", finished_at=_iso(1))

    assert main(["--root", str(root), "--keep", "1"]) == 0
    assert (root / "old_run").is_dir()
    assert (root / "new_run").is_dir()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "old_run" in out


def test_yes_deletes_the_right_ones(tmp_path):
    """--yes removes exactly the runs the policy selected."""
    from scripts.prune_runs import main

    root = tmp_path / "experiments"
    _make_run(root, "r1", finished_at=_iso(10))
    _make_run(root, "r2", finished_at=_iso(5))
    _make_run(root, "r3", finished_at=_iso(1))

    assert main(["--root", str(root), "--keep", "2", "--yes"]) == 0
    assert not (root / "r1").exists()
    assert (root / "r2").is_dir()
    assert (root / "r3").is_dir()


def test_running_run_is_never_deleted(tmp_path):
    """A run in flight is not garbage — it must survive even when it is the oldest."""
    from scripts.prune_runs import main

    root = tmp_path / "experiments"
    _make_run(root, "in_flight", status="running", finished_at=_iso(100))
    _make_run(root, "finished", finished_at=_iso(1))

    assert main(["--root", str(root), "--keep", "1", "--yes"]) == 0
    assert (root / "in_flight").is_dir(), "a 'running' run must never be deleted"
    assert (root / "finished").is_dir()


def test_max_age_keeps_recent_and_drops_old(tmp_path):
    """--max-age protects the recent runs regardless of the keep count."""
    from scripts.prune_runs import main

    root = tmp_path / "experiments"
    _make_run(root, "ancient", finished_at=_iso(400))
    _make_run(root, "yesterday", finished_at=_iso(1))

    assert main(["--root", str(root), "--max-age", "30", "--yes"]) == 0
    assert not (root / "ancient").exists()
    assert (root / "yesterday").is_dir()


def test_keep_ranks_by_timestamp_then_reps(tmp_path):
    """Ties on the timestamp are broken by reps_completed, like the dashboard."""
    from scripts.prune_runs import collect_runs, rank_runs

    root = tmp_path / "experiments"
    stamp = _iso(3)
    _make_run(root, "few_reps", finished_at=stamp, reps=1)
    _make_run(root, "many_reps", finished_at=stamp, reps=9)

    ranked = rank_runs(collect_runs(root))
    assert [r.name for r in ranked][0] == "many_reps"


def test_path_traversal_outside_root_is_refused(tmp_path):
    """A target outside the experiments root must be refused."""
    from scripts.prune_runs import PruneError, delete_run

    root = tmp_path / "experiments"
    root.mkdir()
    outsider = tmp_path / "precious"
    outsider.mkdir()
    (outsider / "keepme.txt").write_text("important", encoding="utf-8")

    with pytest.raises(PruneError):
        delete_run(root, outsider)
    assert (outsider / "keepme.txt").exists()

    # ...and the root itself is refused as well.
    with pytest.raises(PruneError):
        delete_run(root, root)


def test_symlinked_run_is_not_collected(tmp_path):
    """A symlink inside experiments/ pointing outside must not be a deletion candidate."""
    from scripts.prune_runs import collect_runs

    root = tmp_path / "experiments"
    root.mkdir()
    outsider = tmp_path / "precious"
    outsider.mkdir()
    try:
        (root / "linked").symlink_to(outsider, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")

    assert [r.name for r in collect_runs(root)] == []


def test_empty_policy_is_rejected(tmp_path):
    """Without --keep/--max-age the script refuses to act (it would delete everything)."""
    from scripts.prune_runs import main

    root = tmp_path / "experiments"
    _make_run(root, "r1")
    with pytest.raises(SystemExit):
        main(["--root", str(root)])
    assert (root / "r1").is_dir()


def test_corrupt_manifest_does_not_crash(tmp_path, capsys):
    """A corrupt manifest is reported, not fatal — the run stays until a policy says otherwise."""
    from scripts.prune_runs import collect_runs

    root = tmp_path / "experiments"
    run_dir = root / "broken"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text("{not json", encoding="utf-8")

    runs = collect_runs(root)
    assert len(runs) == 1
    assert runs[0].status == "unknown"
