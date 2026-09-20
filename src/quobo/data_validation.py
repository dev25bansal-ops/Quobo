"""Read-only, offline validation of the pipeline's on-disk data contracts.

No pipeline stages are imported: in particular, validation never loads a model,
fetches weights, repairs files, or unpickles an array. NPZ embeddings are checked
in bounded chunks (including Fortran-ordered arrays written by pandas).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.IGNORECASE)
_CHUNK_BYTES = 1024 * 1024


class ValidationReport:
    """JSON-safe errors and counters, accumulated rather than fail-fast."""

    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.warnings: list[str] = []
        self.checks: dict[str, Any] = {}

    def error(self, code: str, path: Any, message: str) -> None:
        self.errors.append({"code": code, "path": str(path), "message": message})

    def as_dict(self) -> dict:
        return {
            "ok": not self.errors,
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _finite(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _portable_parts(value: Any) -> list[str]:
    """Reject Windows as well as POSIX ambiguous/escaping relative paths."""
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive:
        raise ValueError("expected a nonempty relative path without a drive")
    value = value.replace("\\", "/")
    if value.startswith("/"):
        raise ValueError("absolute path is not allowed here")
    parts = value.split("/")
    for part in parts:
        if (
            part in ("", ".", "..")
            or part.endswith((".", " "))
            or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
            or _RESERVED.match(part)
        ):
            raise ValueError("path contains a nonportable or unsafe component")
    return parts


def _contained(root: Path, value: Any, *, absolute: bool = False, base: Path | None = None) -> Path:
    if absolute and isinstance(value, str) and Path(value).is_absolute():
        # Native absolute paths are the actual raw extractor contract. Foreign
        # drives/UNC paths on another OS cannot be verified and must not be remapped.
        candidate = Path(value)
        _portable_parts(candidate.relative_to(candidate.anchor).as_posix())
    else:
        candidate = (base if base is not None else root).joinpath(*_portable_parts(value))
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("resolved path escapes the supplied root")
    return resolved


def _load_json(path: Path, report: ValidationReport) -> dict | None:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=unique, parse_constant=bad_constant)
        if not isinstance(value, dict):
            # Collected as a validation error, not a caller type bug.
            raise ValueError("expected a JSON object")  # noqa: TRY004
        return value
    except (OSError, ValueError, RecursionError) as exc:
        report.error("json", path, str(exc))
        return None


def _records(data: dict, key: str, report: ValidationReport, path: Path) -> list:
    records = data.get(key)
    if not isinstance(records, list) or not records:
        report.error("schema", f"{path}:{key}", "expected a nonempty list")
        return []
    return records


def _index(records: list, name: str, report: ValidationReport) -> dict[int, dict]:
    result = {}
    for i, record in enumerate(records):
        loc = f"{name}[{i}]"
        if not isinstance(record, dict):
            report.error("schema", loc, "expected an object")
        elif not _integer(record.get("id")):
            report.error("id", loc, "id must be a nonnegative integer (not bool)")
        elif record["id"] in result:
            report.error("duplicate_id", loc, f"duplicate id {record['id']}")
        else:
            result[record["id"]] = record
    return result


def _verify_image(path: Path, report: ValidationReport, seen: set[Path]) -> None:
    if path in seen:
        return
    seen.add(path)
    report.checks["images_checked"] = report.checks.get("images_checked", 0) + 1
    try:
        from PIL import Image

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("image has no pixels")
                image.verify()
    except Exception as exc:  # noqa: BLE001 — Pillow plugin errors are collected per image.
        report.error("image", path, f"{type(exc).__name__}: {exc}")


def _scan_images(root: Path, report: ValidationReport, seen: set[Path], *, crops=False) -> None:
    if not root.is_dir():
        report.error("directory", root, "image directory does not exist")
        return
    count = 0
    # Do not follow directory symlinks. Check their targets and report instead.
    import os

    def scan_error(exc):
        report.error("directory", root, str(exc))

    for parent, dirs, files in os.walk(root, followlinks=False, onerror=scan_error):
        for name in dirs[:]:
            child = Path(parent) / name
            if child.is_symlink() or child.resolve() != child.absolute():
                report.error("path", child, "linked image directories are not scanned")
                dirs.remove(name)
        for name in files:
            path = Path(parent) / name
            if not crops and path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                safe = _contained(root, path.relative_to(root).as_posix())
            except (ValueError, OSError, RuntimeError) as exc:
                report.error("path", path, str(exc))
                continue
            if crops and (len(path.relative_to(root).parts) != 2 or path.suffix.lower() != ".jpg"):
                report.error("crop_layout", path, "expected <class>/<filename>.jpg")
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                count += 1
                _verify_image(safe, report, seen)
    if not count:
        report.error("empty_images", root, "no image files were found")


def _validate_coco(
    path: Path, images_dir: Path | None, report: ValidationReport, seen: set[Path]
) -> dict | None:
    data = _load_json(path, report)
    if data is None:
        return None
    categories = _records(data, "categories", report, path)
    images = _records(data, "images", report, path)
    annotations = _records(data, "annotations", report, path)
    cats = _index(categories, "categories", report)
    ims = _index(images, "images", report)
    _index(annotations, "annotations", report)
    report.checks["annotations"] = {
        "categories": len(categories),
        "images": len(images),
        "annotations": len(annotations),
    }
    names = set()
    for i, category in enumerate(categories):
        if not isinstance(category, dict):
            continue
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            report.error("schema", f"categories[{i}].name", "expected a nonempty string")
        elif name in names:
            report.error("category", f"categories[{i}].name", "duplicate category name")
        else:
            names.add(name)
        if "supercategory" in category and not isinstance(category["supercategory"], str):
            report.error("schema", f"categories[{i}].supercategory", "expected a string")
    file_names = set()
    for i, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        for dim in ("width", "height"):
            if not _integer(image.get(dim), 1):
                report.error("dimension", f"images[{i}].{dim}", "expected a positive integer")
        try:
            parts = _portable_parts(image.get("file_name"))
            name = "/".join(parts).casefold()
            if name in file_names:
                report.error("duplicate_path", f"images[{i}]", "duplicate portable image path")
            file_names.add(name)
            if images_dir is not None:
                _verify_image(_contained(images_dir, image["file_name"]), report, seen)
        except (ValueError, OSError, RuntimeError) as exc:
            report.error("path", f"images[{i}].file_name", str(exc))
    for i, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            continue
        loc = f"annotations[{i}]"
        for field, index in (("image_id", ims), ("category_id", cats)):
            ref = annotation.get(field)
            if not _integer(ref) or ref not in index:
                report.error("reference", f"{loc}.{field}", "unknown or invalid integer reference")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(_finite(v) for v in bbox):
            report.error("bbox", loc, "bbox must contain four finite numbers")
        else:
            x, y, w, h = bbox
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                report.error("bbox", loc, "bbox must have nonnegative origin and positive extent")
            image_id = annotation.get("image_id")
            image = ims.get(image_id, {}) if _integer(image_id) else {}
            if (
                _integer(image.get("width"), 1)
                and _integer(image.get("height"), 1)
                and (x + w > image["width"] + 1e-6 or y + h > image["height"] + 1e-6)
            ):
                report.error("bbox", loc, "bbox exceeds original annotation image dimensions")
        if "area" in annotation and (not _finite(annotation["area"]) or annotation["area"] < 0):
            report.error("schema", f"{loc}.area", "expected a nonnegative finite number")
        if "iscrowd" in annotation and (
            type(annotation["iscrowd"]) is not int or annotation["iscrowd"] not in (0, 1)
        ):
            report.error("schema", f"{loc}.iscrowd", "expected integer 0 or 1")
    return data


def _validate_provenance(
    path: Path, annotations: Path | None, coco: dict | None, report: ValidationReport
) -> None:
    data = _load_json(path, report)
    if data is None:
        return
    sha = data.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        report.error("provenance", path, "sha256 must be 64 hexadecimal characters")
    if not isinstance(data.get("source_url"), str) or not data["source_url"].strip():
        report.error("provenance", path, "source_url must be nonempty (never fetched)")
    try:
        timestamp = datetime.fromisoformat(data.get("file_mtime", ""))
        if timestamp.tzinfo is None:
            raise ValueError("file_mtime must include a timezone")
    except (ValueError, TypeError) as exc:
        report.error("provenance", path, f"invalid file_mtime: {exc}")
    for field, key in (("n_images", "images"), ("n_annotations", "annotations")):
        if not _integer(data.get(field), 1):
            report.error("provenance", path, f"{field} must be a positive integer")
        elif coco is not None and isinstance(coco.get(key), list) and data[field] != len(coco[key]):
            report.error("provenance", path, f"{field} differs from annotation snapshot")
    if annotations is not None:
        try:
            digest = hashlib.sha256()
            with annotations.open("rb") as stream:
                for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            report.checks["annotations_sha256"] = actual
            if not isinstance(sha, str) or actual != sha.lower():
                report.error("provenance_sha256", path, "pinned sha256 does not match annotations")
        except OSError as exc:
            report.error("provenance_sha256", annotations, str(exc))


def _crop_inventory(root: Path, classes: list[str], report: ValidationReport) -> tuple[dict, set]:
    """Filename identity of the crop set, via features.crops_fingerprint (M-05:
    the formula is defined in exactly one place — this wrapper only adds the
    path-containment checks and error reporting the validator needs).

    This is a filename identity, NOT a content digest; the test suite compares
    it with the real extractor function.
    """
    from .features import crops_fingerprint

    paths = set()
    for cls in classes:
        try:
            cls_dir = _contained(root, cls)
            if not cls_dir.is_dir():
                continue
            for _p in cls_dir.glob("*.jpg"):
                paths.add(_p)
        except (OSError, ValueError, RuntimeError) as exc:
            report.error("path", root / cls, str(exc))
    fp = crops_fingerprint(root, classes)
    if fp["n_crops"] == 0:
        report.error("empty_crops", root, "no crops for the explicitly supplied classes")
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and child.name not in classes:
                report.error("crop_class", child, "class directory is absent from --classes")
    report.checks["crops_fingerprint"] = fp
    return fp, paths


def _counts(data: Any, loc: str, report: ValidationReport) -> dict | None:
    if not isinstance(data, dict) or any(
        not isinstance(k, str) or not k or not _integer(v) for k, v in data.items()
    ):
        report.error("metadata", loc, "expected string keys and nonnegative integer counts")
        return None
    return data


def _validate_crop_meta(path: Path, fp: dict | None, report: ValidationReport) -> None:
    data = _load_json(path, report)
    if data is None:
        return
    maps = {
        k: _counts(data.get(k), f"{path}:{k}", report)
        for k in ("per_class", "kept_classes", "removed_classes", "skipped")
    }
    kept, removed, counts = maps["kept_classes"], maps["removed_classes"], maps["per_class"]
    if kept is not None and removed is not None and counts is not None:
        if set(kept) & set(removed):
            report.error("metadata", path, "kept and removed classes overlap")
        for cls, count in {**kept, **removed}.items():
            # build_crops Counter omits classes that produced zero crops.
            if counts.get(cls, 0) != count:
                report.error("metadata", path, f"per_class count disagrees for {cls}")
        if set(counts) - (set(kept) | set(removed)):
            report.error("metadata", path, "per_class contains an unaccounted class")
        if fp is not None:
            actual = {k: v for k, v in fp["per_class"].items() if v}
            if actual != {k: v for k, v in kept.items() if v}:
                report.error("crop_counts", path, "kept_classes differs from crops on disk")
    report.checks["crop_metadata"] = str(path)


def _read_npz(
    path: Path, report: ValidationReport, expected_dim: int, max_bytes: int
) -> dict[str, Any]:
    """Read NPY headers and stream payloads; never np.load the embedding matrix."""
    import numpy as np
    from numpy.lib import format as fmt

    arrays: dict[str, Any] = {}
    required = {"embeddings.npy", "labels.npy", "crop_paths.npy"}
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if Counter(i.filename for i in entries) != Counter(required):
                report.error(
                    "cache_members", path, "expected exactly embeddings, labels, crop_paths"
                )
            if sum(i.file_size for i in entries) > max_bytes:
                report.error(
                    "cache_size", path, f"uncompressed cache exceeds {max_bytes} byte limit"
                )
                return arrays
            for entry in entries:
                if entry.filename not in required:
                    continue
                loc = f"{path}:{entry.filename}"
                try:
                    with archive.open(entry) as stream:
                        version = fmt.read_magic(stream)
                        if version == (1, 0):
                            shape, _, dtype = fmt.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            shape, _, dtype = fmt.read_array_header_2_0(stream)
                        else:
                            raise ValueError(f"unsupported NPY version {version}")
                        key = entry.filename[:-4]
                        report.checks.setdefault("cache_arrays", {})[key] = {
                            "shape": list(shape),
                            "dtype": str(dtype),
                        }
                        ndim = 2 if key == "embeddings" else 1
                        if len(shape) != ndim or any(n <= 0 for n in shape):
                            raise ValueError(f"expected a nonempty {ndim}-dimensional array")
                        if key == "embeddings" and shape[1] != expected_dim:
                            report.error(
                                "cache_dimension",
                                loc,
                                f"expected embedding width {expected_dim}, got {shape[1]}",
                            )
                        expected_kind = "f" if key == "embeddings" else "U"
                        if dtype.hasobject or dtype.kind != expected_kind or dtype.itemsize <= 0:
                            raise ValueError(
                                f"expected dtype kind {expected_kind}; pickle is forbidden"
                            )
                        size = math.prod(shape) * dtype.itemsize
                        if size != entry.file_size - stream.tell():
                            raise ValueError("NPY shape/dtype and actual payload size disagree")
                        step = max(1, _CHUNK_BYTES // dtype.itemsize) * dtype.itemsize
                        values = []
                        nonfinite = False
                        for remaining in range(size, 0, -step):
                            payload = stream.read(min(step, remaining))
                            if len(payload) != min(step, remaining):
                                raise ValueError("truncated array payload")
                            block = np.frombuffer(payload, dtype=dtype)
                            if key == "embeddings":
                                nonfinite |= not bool(np.isfinite(block).all())
                            else:
                                values.extend(block.tolist())
                        if nonfinite:
                            report.error("cache_finite", loc, "embeddings contain NaN or infinity")
                        arrays[key] = shape if key == "embeddings" else values
                except (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
                    report.error("cache_array", loc, str(exc))
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        report.error("cache", path, str(exc))
    return arrays


def _validate_cache(
    path: Path,
    meta_path: Path | None,
    root: Path | None,
    classes: list[str],
    fp: dict | None,
    expected_paths: set,
    report: ValidationReport,
    expected_dim: int,
    max_bytes: int,
    path_base: Path,
) -> None:
    arrays = _read_npz(path, report, expected_dim, max_bytes)
    meta = _load_json(meta_path, report) if meta_path is not None else None
    if meta is not None:
        if not _integer(meta.get("n_samples"), 1):
            report.error("metadata", meta_path, "n_samples must be a positive integer")
        cached = meta.get("crops_fingerprint")
        if not isinstance(cached, dict):
            report.error("fingerprint", meta_path, "crops_fingerprint object is required")
        else:
            _counts(cached.get("per_class"), f"{meta_path}:per_class", report)
            if (
                not _integer(cached.get("n_crops"), 1)
                or not isinstance(cached.get("files_hash"), str)
                or not re.fullmatch(r"[0-9a-f]{16}", cached["files_hash"])
            ):
                report.error("fingerprint", meta_path, "invalid n_crops or 16-character files_hash")
            if fp is not None and cached != fp:
                report.error(
                    "fingerprint",
                    meta_path,
                    "cache fingerprint differs from extractor's ordered crop inventory",
                )
        if "embeddings" in arrays and meta.get("n_samples") != arrays["embeddings"][0]:
            report.error("cache_rows", meta_path, "n_samples differs from embedding rows")
    n = arrays.get("embeddings", (None,))[0]
    for key in ("labels", "crop_paths"):
        if key in arrays and n is not None and len(arrays[key]) != n:
            report.error("cache_rows", path, f"{key} length differs from embedding rows")
    labels = arrays.get("labels", [])
    for i, label in enumerate(labels):
        if not label or label not in classes:
            report.error("cache_label", f"{path}:labels[{i}]", "label absent from --classes")
    if fp is not None and Counter(labels) != Counter(
        {k: v for k, v in fp["per_class"].items() if v}
    ):
        report.error("cache_labels", path, "label counts differ from crop inventory")
    actual_paths = set()
    for i, value in enumerate(arrays.get("crop_paths", [])):
        try:
            if root is None:
                _portable_parts(value) if not Path(value).is_absolute() else None
                continue
            crop = _contained(root, value, absolute=True, base=path_base)
            if crop in actual_paths:
                report.error("duplicate_path", f"{path}:crop_paths[{i}]", "duplicate crop path")
            actual_paths.add(crop)
            if not crop.is_file():
                report.error("cache_path", crop, "cached crop does not exist")
            if i < len(labels) and crop.parent != (root / labels[i]).resolve():
                report.error("cache_label", crop, "label does not match crop class directory")
        except (ValueError, OSError, RuntimeError) as exc:
            report.error("cache_path", f"{path}:crop_paths[{i}]", str(exc))
    if root is not None and actual_paths != expected_paths:
        report.error("cache_paths", path, "cached crop path set differs from crop inventory")
    report.warnings.append(
        "Crop fingerprints identify ordered filenames/counts, not image bytes; "
        "matching fingerprints do not authenticate crop contents."
    )


def validate_data(
    *,
    annotations: Path | None = None,
    images_dir: Path | None = None,
    provenance: Path | None = None,
    crops_dir: Path | None = None,
    crop_meta: Path | None = None,
    classes: list[str] | None = None,
    cache: Path | None = None,
    cache_meta: Path | None = None,
    embedding_dim: int = 1280,
    max_cache_bytes: int = 2 * 1024**3,
    cache_path_base: Path | None = None,
) -> dict:
    """Validate every explicitly supplied modality and return a JSON-safe report.

    Annotation-only validation checks the COCO fields consumed by build_crops,
    not segmentation polygons. Supply images_dir for all-image Pillow verification.
    Decoded image dimensions may differ from COCO originals (Flickr 640px contract).
    A cache requires its metadata, crops root and ordered classes. No defaults
    silently point at the user's real dataset.
    """
    report = ValidationReport()
    supplied = {
        "annotations": annotations,
        "images_dir": images_dir,
        "provenance": provenance,
        "crops_dir": crops_dir,
        "crop_meta": crop_meta,
        "cache": cache,
        "cache_meta": cache_meta,
    }
    supplied = {k: Path(v) for k, v in supplied.items() if v is not None}
    report.checks["requested"] = {k: str(v) for k, v in supplied.items()}
    if not supplied:
        report.error("arguments", "input", "supply at least one data modality")
    for key, dependency in (
        ("provenance", "annotations"),
        ("crop_meta", "crops_dir"),
        ("cache", "cache_meta"),
        ("cache_meta", "cache"),
        ("cache", "crops_dir"),
    ):
        if key in supplied and dependency not in supplied:
            report.error("arguments", key, f"requires --{dependency.replace('_', '-')}")
    classes = classes or []
    good_classes = []
    for cls in classes:
        try:
            if len(_portable_parts(cls)) != 1:
                raise ValueError("class must be a single portable directory name")
            if cls.casefold() in {c.casefold() for c in good_classes}:
                raise ValueError("duplicate class (case-insensitive)")
            good_classes.append(cls)
        except ValueError as exc:
            report.error("arguments", "classes", str(exc))
    if (crops_dir is not None or cache is not None) and not good_classes:
        report.error("arguments", "classes", "supply the extractor's ordered --classes")
    if not _integer(embedding_dim, 1) or not _integer(max_cache_bytes, 1):
        report.error("arguments", "cache", "embedding_dim and max_cache_bytes must be positive")
        return report.as_dict()
    seen: set[Path] = set()
    coco = None
    if annotations is not None:
        coco = _validate_coco(
            Path(annotations), Path(images_dir) if images_dir else None, report, seen
        )
    if images_dir is not None:
        _scan_images(Path(images_dir), report, seen)
    if provenance is not None:
        _validate_provenance(
            Path(provenance), Path(annotations) if annotations else None, coco, report
        )
    fp, paths = None, set()
    if crops_dir is not None:
        _scan_images(Path(crops_dir), report, seen, crops=True)
        fp, paths = _crop_inventory(Path(crops_dir), good_classes, report)
    if crop_meta is not None:
        _validate_crop_meta(Path(crop_meta), fp, report)
    if cache is not None:
        _validate_cache(
            Path(cache),
            Path(cache_meta) if cache_meta else None,
            Path(crops_dir) if crops_dir else None,
            good_classes,
            fp,
            paths,
            report,
            embedding_dim,
            max_cache_bytes,
            Path(cache_path_base or Path.cwd()),
        )
    elif cache_meta is not None:
        _load_json(Path(cache_meta), report)  # supplied malformed metadata is still reported
    return report.as_dict()
