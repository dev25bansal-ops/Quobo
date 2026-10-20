"""Leak audit: prove the leak-free claim from a run's persisted artifacts.

The paper's central methodological claim is that the protocol is *leak-free*:
the scaler and PCA are refit inside every train/test split, and no source
photo's crops straddle the split. Until now that claim was structural — it
followed from reading the code and from a counter the runner wrote into the
manifest (``group_overlap_events``).

This module makes the claim *per-artifact and independently checkable*. It
re-derives the evidence from the artifacts a run left on disk, so a reviewer
can audit a published run without trusting the code that produced it:

- **Group overlap.** Recompute, from the persisted held-out crop ids of every
  repeat, how many source photos appear on BOTH sides of that repeat's split.
  A photo contributing several crops is normal; a photo with one crop held
  out and another in train is exactly the leak that the group-aware split
  exists to prevent. The count is over *source photos*, matching the
  semantics of the runner's own ``group_overlap_events`` counter, so the two
  can be compared like for like.

- **Held-out uniqueness.** Verify that within a repeat the held-out crop IDs
  are unique, and that the recorded ``y_test`` and each classifier's
  predictions are the same length as the held-out set (a misaligned artifact
  would silently corrupt the dashboard's vote aggregation).

- **Shared split across arms.** Every arm of a repeat must be evaluated on
  the SAME held-out set. If two arms saw different crops, the paired
  significance tests would be comparing different data and the comparison
  would be meaningless.

- **Group-label integrity.** The ``test_groups`` the runner recorded must
  agree with the group derived from the crop id itself. A disagreement means
  the artifact's group labels cannot be trusted to reason about leakage.

- **Manifest agreement.** Compare the recomputed overlap count against the
  manifest's ``group_overlap_events`` when the manifest records one, and
  report a mismatch rather than trusting either side.

- **Hash integrity.** Re-verify every artifact's sha256 against the manifest
  inventory, so the audit cannot be fooled by a substituted file.

The audit is READ-ONLY and OFFLINE: it never retrains, never replays a split,
and never writes into the run directory. A run whose manifest is missing or
not ``complete`` is refused, on the same principle as the dashboard — an
incomplete run is not evidence of anything.

What it deliberately does NOT do: recompute the split. Re-deriving the split
from the seed would *replay* it, which is exactly what the provenance rules
forbid — the persisted crop IDs are the authority. The audit reads the
recorded membership; it never regenerates it.

One honest limitation, surfaced rather than hidden: the *train* side of a
split is not persisted. The audit reconstructs it from the crop inventory on
disk, so it can only prove the absence of straddling for photos that are
still present on disk. If the inventory is missing or incomplete the audit
reports that the straddle check could not be performed — it never reports
"no overlap" on the strength of an inventory it could not read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_NAME = "run_manifest.json"
CONFIG_NAME = "config_used.json"
DEFAULT_CROPS_SUBDIR = Path("data") / "processed" / "crops"


class LeakAuditError(RuntimeError):
    """A run cannot be audited: missing/corrupt manifest, or an incomplete run."""


@dataclass
class RepeatAudit:
    """Audit result for one repeat of a run."""

    rep: int
    arms: list[str] = field(default_factory=list)
    n_test: int = 0
    n_groups: int = 0
    #: source photos that appear in this repeat's held-out set more than once
    #: (expected: a photo contributes several crops)
    groups_with_multiple_crops: int = 0
    #: held-out crop ids duplicated within the repeat (must be zero)
    duplicate_test_ids: int = 0
    #: arms whose y_test or prediction vectors are not the same length as the
    #: held-out set (must be zero)
    misaligned_artifacts: int = 0
    #: hash mismatches against the manifest inventory (must be zero)
    hash_mismatches: int = 0
    #: arms whose held-out crop ids differ from the first arm's for the same
    #: repeat — the split must be shared by every arm of a repeat
    arms_with_different_split: list[str] = field(default_factory=list)
    #: held-out crops whose recorded group label disagrees with the group
    #: derived from the crop id (must be zero)
    group_label_mismatches: int = 0
    #: held-out crop ids that are no longer present in the crop inventory
    held_out_crops_missing: int = 0
    #: source photos that have at least one crop held out AND at least one
    #: crop on the reconstructed train side (the leak being audited for)
    straddling_groups: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.duplicate_test_ids == 0
            and self.misaligned_artifacts == 0
            and self.hash_mismatches == 0
            and not self.arms_with_different_split
            and self.group_label_mismatches == 0
            and self.held_out_crops_missing == 0
            and self.straddling_groups == 0
        )

    def as_dict(self) -> dict:
        return {
            "rep": self.rep,
            "arms": self.arms,
            "n_test": self.n_test,
            "n_groups": self.n_groups,
            "groups_with_multiple_crops": self.groups_with_multiple_crops,
            "duplicate_test_ids": self.duplicate_test_ids,
            "misaligned_artifacts": self.misaligned_artifacts,
            "hash_mismatches": self.hash_mismatches,
            "arms_with_different_split": self.arms_with_different_split,
            "group_label_mismatches": self.group_label_mismatches,
            "held_out_crops_missing": self.held_out_crops_missing,
            "straddling_groups": self.straddling_groups,
            "ok": self.ok,
        }


@dataclass
class RunAudit:
    """Audit result for a whole run."""

    run_id: str
    run_dir: str
    status: str
    reps_completed: int
    repeats: list[RepeatAudit] = field(default_factory=list)
    #: number of repeat pairs whose held-out sets share at least one source
    #: photo. NOT leakage — repeats deliberately reuse the photo pool with
    #: different splits. Reported so a reader can see the repeats are
    #: genuinely different splits rather than one split repeated.
    repeats_sharing_photos: int = 0
    manifest_group_overlap_events: int | None = None
    manifest_overlap_agrees: bool | None = None
    #: whether the train side could be reconstructed well enough to check for
    #: straddling photos
    straddle_check_performed: bool = False
    crops_root: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(r.ok for r in self.repeats)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "reps_completed": self.reps_completed,
            "n_repeats_audited": len(self.repeats),
            "repeats_sharing_photos": self.repeats_sharing_photos,
            "manifest_group_overlap_events": self.manifest_group_overlap_events,
            "manifest_overlap_agrees": self.manifest_overlap_agrees,
            "straddle_check_performed": self.straddle_check_performed,
            "crops_root": self.crops_root,
            "ok": self.ok,
            "errors": list(self.errors),
            "repeats": [r.as_dict() for r in self.repeats],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LeakAuditError(f"{path.name} is not valid JSON: {exc}") from exc


def _manifest(run_dir: Path) -> dict:
    path = run_dir / MANIFEST_NAME
    if not path.is_file():
        raise LeakAuditError(
            f"run {run_dir.name!r} has no {MANIFEST_NAME}; an unauditable run is not "
            "evidence — re-run the experiment to write one"
        )
    manifest = _read_json(path)
    status = manifest.get("status")
    if status != "complete":
        raise LeakAuditError(
            f"run {run_dir.name!r} status is {status!r}, not 'complete'; "
            "refusing to audit a partial run"
        )
    reps = manifest.get("reps_completed")
    if not isinstance(reps, int) or reps < 1:
        raise LeakAuditError(f"run {run_dir.name!r} manifest records no completed repeats")
    return manifest


def _group_of(crop_id: str) -> str:
    """Source-photo group key of a crop id.

    Mirrors ``run_experiment.crop_image_group`` (everything before the final
    ``_annNNNN``) but works from the persisted crop ID, which may be an
    absolute Windows path. The audit must not import the pipeline's grouping
    helper: an audit that reuses the audited code cannot catch a bug in it.
    """
    name = str(crop_id).replace("\\", "/").split("/")[-1]
    return name.rsplit("_ann", 1)[0]


def _basename(crop_id: str) -> str:
    return str(crop_id).replace("\\", "/").split("/")[-1]


def _resolve_crops_root(run_dir: Path, manifest: dict) -> Path | None:
    """Locate the crop inventory the run drew from.

    The run records the config it used, including ``data.crops_dir_override``;
    that is the authoritative location. Without an override the pipeline uses
    ``<repo>/data/processed/crops``, and the repo root is the parent of the
    ``experiments/`` directory the run lives in.
    """
    config_path = run_dir / CONFIG_NAME
    if not config_path.is_file():
        config_path = None
    else:
        try:
            cfg = _read_json(config_path)
        except LeakAuditError:
            cfg = {}
        override = ((cfg.get("data") or {}).get("crops_dir_override")) or ""
        if override:
            override_path = Path(override)
            repo_root = run_dir.parent.parent
            return override_path if override_path.is_absolute() else repo_root / override_path

    repo_root = run_dir.parent.parent
    default = repo_root / DEFAULT_CROPS_SUBDIR
    if default.is_dir():
        return default
    return None


def _load_inventory(crops_root: Path) -> dict[str, set[str]]:
    """Map source-photo group -> set of crop file names present on disk."""
    inventory: dict[str, set[str]] = {}
    for class_dir in sorted(crops_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in class_dir.glob("*.jpg"):
            inventory.setdefault(_group_of(path.name), set()).add(path.name)
    return inventory


def audit_repeat(
    rep: int,
    run_dir: Path,
    arms: list[str],
    inventory: dict,
    payloads: dict[str, dict] | None = None,
) -> RepeatAudit:
    """Audit one repeat's artifacts.

    ``inventory`` maps source-photo group -> set of crop file names on disk.
    ``payloads`` optionally supplies already-parsed artifact payloads keyed by
    file name, so a run is not read from disk twice.
    """
    result = RepeatAudit(rep=rep, arms=list(arms))
    reference_ids: list[str] | None = None
    # precomputed once: the flat set of every crop present on disk
    inventory_names: set[str] = set()
    for group_names in inventory.values():
        inventory_names |= group_names

    for arm in arms:
        name = f"rep{rep}_{arm}.json"
        path = run_dir / name
        if not path.is_file():
            result.misaligned_artifacts += 1
            result.hash_mismatches += 1
            continue
        payload = (payloads or {}).get(name)
        if payload is None:
            payload = _read_json(path)

        test_ids = [str(t) for t in (payload.get("test_ids") or [])]
        y_test = payload.get("y_test") or []
        groups = [str(g) for g in (payload.get("test_groups") or [])]

        if reference_ids is None:
            reference_ids = list(test_ids)
            result.n_test = len(test_ids)
            result.n_groups = len(set(groups))
            result.duplicate_test_ids = len(test_ids) - len(set(test_ids))
            counts: dict[str, int] = {}
            for g in groups:
                counts[g] = counts.get(g, 0) + 1
            result.groups_with_multiple_crops = sum(1 for v in counts.values() if v > 1)

            # the recorded group label must agree with the group implied by
            # the crop id, otherwise the artifact cannot be reasoned about
            if len(groups) == len(test_ids):
                for crop_id, group in zip(test_ids, groups):
                    if _group_of(crop_id) != group:
                        result.group_label_mismatches += 1

            # reconstruct the train side from the inventory and look for
            # source photos that straddle the split
            if inventory:
                held_out = {_basename(t) for t in test_ids}
                held_groups = {_group_of(t) for t in test_ids}
                missing = sum(1 for t in test_ids if _basename(t) not in inventory_names)
                result.held_out_crops_missing = missing
                straddling = 0
                for group, names in inventory.items():
                    if group not in held_groups:
                        continue
                    if any(name not in held_out for name in names):
                        straddling += 1
                result.straddling_groups = straddling
        elif list(test_ids) != reference_ids:
            # Every arm of a repeat must be evaluated on the SAME held-out set;
            # otherwise the paired significance tests compare different data.
            result.arms_with_different_split.append(arm)

        if len(y_test) != len(test_ids):
            result.misaligned_artifacts += 1
        if groups and len(groups) != len(test_ids):
            result.misaligned_artifacts += 1
        metrics = payload.get("metrics") or {}
        for _clf, m in metrics.items():
            preds = (m or {}).get("predictions")
            if preds is None or len(preds) != len(test_ids):
                result.misaligned_artifacts += 1
    return result


def audit_run(run_dir: Path, crops_root: Path | None = None) -> RunAudit:
    """Audit a completed run directory and return a structured verdict.

    Raises :class:`LeakAuditError` when the run cannot be audited at all
    (missing/corrupt manifest, incomplete status, no completed repeats).
    """
    run_dir = Path(run_dir)
    manifest = _manifest(run_dir)
    arms = list(manifest.get("arms") or [])
    reps_completed = int(manifest["reps_completed"])
    artifact_inventory = manifest.get("artifacts") or {}

    audit = RunAudit(
        run_id=str(manifest.get("run_id") or run_dir.name),
        run_dir=str(run_dir),
        status=str(manifest.get("status")),
        reps_completed=reps_completed,
    )

    # resolve the crop inventory used to reconstruct the train side
    root = crops_root if crops_root is not None else _resolve_crops_root(run_dir, manifest)
    inventory: dict[str, set[str]] = {}
    if root is not None and Path(root).is_dir():
        inventory = _load_inventory(Path(root))
        audit.crops_root = str(root)
        audit.straddle_check_performed = bool(inventory)
    else:
        audit.errors.append(
            "the crop inventory could not be located, so the train side of each "
            "split could not be reconstructed; the absence of straddling source "
            "photos could NOT be verified for this run"
        )

    # read each artifact once and reuse it for the per-repeat audit and for
    # the cross-repeat comparison
    payloads: dict[str, dict] = {}
    hash_mismatches: dict[int, int] = {}
    for rep in range(reps_completed):
        for arm in arms:
            name = f"rep{rep}_{arm}.json"
            path = run_dir / name
            if not path.is_file():
                continue
            recorded = artifact_inventory.get(name)
            if not recorded or _sha256_file(path) != recorded.get("sha256"):
                hash_mismatches[rep] = hash_mismatches.get(rep, 0) + 1
            try:
                payloads[name] = _read_json(path)
            except LeakAuditError:
                continue

    groups_by_rep: dict[int, set[str]] = {}
    for rep in range(reps_completed):
        result = audit_repeat(rep, run_dir, arms, inventory, payloads=payloads)
        result.hash_mismatches += hash_mismatches.get(rep, 0)
        audit.repeats.append(result)
        if not result.ok:
            audit.errors.append(f"repeat {rep} failed the audit")
        groups_by_rep[rep] = set()

    # Recompute the overlap between repeats: a source photo that is held out
    # in one repeat and also held out in another is NOT leakage (repeats
    # deliberately reuse the photo pool with different splits). This is
    # reported as context so a reader can see the repeats are genuinely
    # different splits rather than one split repeated.
    for rep in range(reps_completed):
        for arm in arms:
            payload = payloads.get(f"rep{rep}_{arm}.json")
            if payload is None:
                continue
            for crop_id in payload.get("test_ids") or []:
                groups_by_rep[rep].add(_group_of(crop_id))
    sharing = sum(
        1
        for a in range(reps_completed)
        for b in range(a + 1, reps_completed)
        if groups_by_rep[a] & groups_by_rep[b]
    )
    audit.repeats_sharing_photos = sharing

    recorded = manifest.get("group_overlap_events")
    if isinstance(recorded, int):
        audit.manifest_group_overlap_events = recorded
        # The manifest counter is the number of (source photo, repeat) pairs
        # that straddled train/test WITHIN a repeat. The audit recomputes the
        # same quantity from the persisted artifacts plus the crop inventory,
        # so the two are directly comparable.
        straddling = sum(r.straddling_groups for r in audit.repeats)
        audit.manifest_overlap_agrees = straddling == recorded
        if straddling != recorded:
            audit.errors.append(
                f"manifest group_overlap_events={recorded} but the artifacts imply "
                f"{straddling}; the manifest and the per-repeat artifacts disagree"
            )
    return audit


def format_report(audit: RunAudit) -> str:
    """Human-readable audit report."""
    lines = [
        f"Leak audit — run {audit.run_id}",
        f"  status                 : {audit.status} ({audit.reps_completed} repeats)",
        f"  repeats audited        : {len(audit.repeats)}",
        f"  repeats sharing photos : {audit.repeats_sharing_photos} (expected > 0: "
        "repeats reuse the photo pool with different splits)",
    ]
    if audit.straddle_check_performed:
        lines.append(f"  crop inventory          : {audit.crops_root}")
        lines.append(
            "  straddling photos       : "
            f"{sum(r.straddling_groups for r in audit.repeats)} "
            "(must be 0 — no source photo may appear on both sides of a split)"
        )
    else:
        lines.append("  straddling photos       : NOT CHECKED (crop inventory unavailable)")
    if audit.manifest_group_overlap_events is not None:
        agrees = audit.manifest_overlap_agrees
        lines.append(
            f"  manifest overlap        : {audit.manifest_group_overlap_events} "
            f"(agrees with artifacts: {agrees})"
        )
    else:
        lines.append(
            "  manifest overlap        : not recorded by this run "
            "(predates the counter; the artifacts were audited directly)"
        )
    lines.append(f"  verdict                 : {'PASS' if audit.ok else 'FAIL'}")
    for rep in audit.repeats:
        if rep.ok:
            continue
        lines.append(
            f"    repeat {rep.rep}: dup_ids={rep.duplicate_test_ids} "
            f"misaligned={rep.misaligned_artifacts} hash_mismatch={rep.hash_mismatches} "
            f"split_differs={rep.arms_with_different_split} "
            f"group_mismatch={rep.group_label_mismatches} "
            f"missing_crops={rep.held_out_crops_missing} "
            f"straddling={rep.straddling_groups}"
        )
    for err in audit.errors:
        lines.append(f"  ! {err}")
    return "\n".join(lines)
