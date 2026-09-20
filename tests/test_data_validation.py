"""Offline validator tests: tiny local files only; no dataset or model downloads."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.data_validation import validate_data

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def codes(report):
    return {error["code"] for error in report["errors"]}


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    (images / "batch_1").mkdir(parents=True)
    # Resized source images deliberately differ from original COCO dimensions.
    Image.new("RGB", (12, 10), "red").save(images / "batch_1" / "source.jpg")
    coco = {
        "categories": [{"id": 0, "name": "Glass bottle", "supercategory": "Bottle"}],
        "images": [{"id": 0, "width": 120, "height": 100, "file_name": "batch_1/source.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 0,
                "category_id": 0,
                "bbox": [10.0, 5.0, 40.0, 30.0],
                "area": 1200.0,
                "iscrowd": 0,
            }
        ],
    }
    annotations = dump(tmp_path / "annotations.json", coco)
    provenance = dump(
        tmp_path / "provenance.json",
        {
            "sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
            "source_url": "https://example.invalid/never-fetched",
            "file_mtime": "2026-01-01T00:00:00+00:00",
            "n_images": 1,
            "n_annotations": 1,
        },
    )
    crops = tmp_path / "crops"
    classes = ["can", "bottle"]  # Deliberately NOT alphabetical; fingerprint honors this order.
    paths = []
    for cls in classes:
        (crops / cls).mkdir(parents=True)
        for name in ("b.jpg", "a.jpg"):
            path = crops / cls / name
            Image.new("RGB", (8, 9), cls == "can" and "blue" or "red").save(path)
        paths.extend(str(crops / cls / name) for name in ("a.jpg", "b.jpg"))
    fp = {
        "per_class": {cls: 2 for cls in classes},
        "n_crops": 4,
        "files_hash": hashlib.sha256(
            "\n".join(f"{cls}/{name}" for cls in classes for name in ("a.jpg", "b.jpg")).encode()
        ).hexdigest()[:16],
    }
    meta = dump(tmp_path / "raw_meta.json", {"n_samples": 4, "crops_fingerprint": fp})
    crop_meta = dump(
        tmp_path / "stats.json",
        {
            "per_class": fp["per_class"],
            "kept_classes": fp["per_class"],
            "removed_classes": {},
            "skipped": {"too_small": 1},
        },
    )
    arrays = {
        "embeddings": np.asfortranarray(np.arange(12, dtype=np.float32).reshape(4, 3)),
        "labels": np.array(["can", "can", "bottle", "bottle"]),
        "crop_paths": np.array(paths),
    }
    cache = tmp_path / "raw.npz"
    np.savez_compressed(cache, **arrays)
    args = {
        "annotations": annotations,
        "images_dir": images,
        "provenance": provenance,
        "crops_dir": crops,
        "classes": classes,
        "crop_meta": crop_meta,
        "cache": cache,
        "cache_meta": meta,
        "embedding_dim": 3,
    }
    return args, coco, arrays


def test_all_supplied_modalities_validated_without_writes(dataset):
    args, _, _ = dataset
    root = args["annotations"].parent
    before = {
        str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in root.rglob("*")
        if p.is_file()
    }
    report = validate_data(**args)
    assert report["ok"], report
    assert report["checks"]["images_checked"] == 5
    assert report["checks"]["cache_arrays"]["embeddings"]["shape"] == [4, 3]
    after = {
        str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_fingerprint_matches_real_extractor_contract(dataset):
    pytest.importorskip("tensorflow")
    from src.quobo.features import crops_fingerprint

    args, _, _ = dataset
    report = validate_data(**args)
    assert report["checks"]["crops_fingerprint"] == crops_fingerprint(
        args["crops_dir"], args["classes"]
    )
    # Reordering class arguments is not silently sorted to the tag-name order.
    args["classes"] = list(reversed(args["classes"]))
    assert "fingerprint" in codes(validate_data(**args))


@pytest.mark.parametrize(
    "path",
    [
        "../outside.jpg",
        "folder/../file.jpg",
        "..\\outside.jpg",
        "/absolute.jpg",
        "C:\\outside.jpg",
        "C:outside.jpg",
        "\\\\server\\share\\file.jpg",
        "folder//file.jpg",
        "folder/NUL.jpg",
        "folder/a.jpg:stream",
        "folder./a.jpg",
        "folder/a?.jpg",
        "folder/a\x00.jpg",
    ],
)
def test_portable_annotation_paths_rejected(dataset, path):
    args, coco, _ = dataset
    coco["images"][0]["file_name"] = path
    dump(args["annotations"], coco)
    report = validate_data(annotations=args["annotations"], images_dir=args["images_dir"])
    assert not report["ok"] and "path" in codes(report)


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda d: d["categories"].append(d["categories"][0].copy()), "duplicate_id"),
        (lambda d: d["images"].append(d["images"][0].copy()), "duplicate_id"),
        (lambda d: d["annotations"].append(d["annotations"][0].copy()), "duplicate_id"),
        (lambda d: d["images"][0].update(width=True), "dimension"),
        (lambda d: d["images"][0].update(height=0), "dimension"),
        (lambda d: d["images"][0].update(width="120"), "dimension"),
        (lambda d: d["annotations"][0].update(image_id=[]), "reference"),
        (lambda d: d["annotations"][0].update(category_id=True), "reference"),
        (lambda d: d["annotations"][0].update(category_id=99), "reference"),
        (lambda d: d["annotations"][0].update(id=-1), "id"),
        (lambda d: d["annotations"][0].update(bbox=[0, 0, 0, 1]), "bbox"),
        (lambda d: d["annotations"][0].update(bbox=[-1, 0, 2, 1]), "bbox"),
        (lambda d: d["annotations"][0].update(bbox=[119, 0, 2, 1]), "bbox"),
        (lambda d: d["annotations"][0].update(bbox=[0, 0, "2", 1]), "bbox"),
        (lambda d: d["annotations"][0].update(bbox=[0, 0, True, 1]), "bbox"),
        (lambda d: d["annotations"][0].update(area=-1), "schema"),
        (lambda d: d["annotations"][0].update(iscrowd=True), "schema"),
        (lambda d: d.update(categories={}), "schema"),
        (lambda d: d.update(annotations=[]), "schema"),
        (lambda d: d["categories"][0].update(name=""), "schema"),
    ],
)
def test_coco_schema_and_references(dataset, change, code):
    args, coco, _ = dataset
    change(coco)
    dump(args["annotations"], coco)
    report = validate_data(annotations=args["annotations"])
    assert not report["ok"] and code in codes(report), report


def test_collects_image_failures_and_scans_unreferenced_images(dataset):
    args, _, _ = dataset
    (args["images_dir"] / "batch_1/source.jpg").write_bytes(b"not an image")
    (args["images_dir"] / "extra.png").write_bytes(b"also not an image")
    (args["crops_dir"] / "can/a.jpg").write_bytes(b"bad crop")
    report = validate_data(**args)
    assert len([e for e in report["errors"] if e["code"] == "image"]) == 3
    assert report["checks"]["images_checked"] == 6
    assert "cache_arrays" in report["checks"]  # No early exit after unrelated image failure.


def test_pillow_verify_is_called_for_every_image(dataset, monkeypatch):
    args, _, _ = dataset
    from PIL import JpegImagePlugin

    calls = []
    original = JpegImagePlugin.JpegImageFile.verify

    def verify(image):
        calls.append(image.filename)
        return original(image)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "verify", verify)
    assert validate_data(**args)["ok"]
    assert len(calls) == 5


@pytest.mark.parametrize(
    "text", ["{", "[]", '{"images": [], "images": []}', '{"value": NaN}', '{"value": Infinity}']
)
def test_bad_json_fails_explicitly(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    report = validate_data(annotations=path)
    assert not report["ok"] and "json" in codes(report)


def test_nonfinite_bbox_and_extreme_integer_fail_without_crash(dataset):
    args, coco, _ = dataset
    coco["annotations"][0]["bbox"] = [0, 0, 10**400, 1]
    dump(args["annotations"], coco)
    assert not validate_data(annotations=args["annotations"])["ok"]
    args["annotations"].write_text(
        json.dumps(coco).replace(str(10**400), "1e400"), encoding="utf-8"
    )
    assert not validate_data(annotations=args["annotations"])["ok"]


def test_provenance_detects_changed_snapshot_and_counts(dataset):
    args, coco, _ = dataset
    coco["images"][0]["width"] = 121
    dump(args["annotations"], coco)
    meta = json.loads(args["provenance"].read_text())
    meta["n_images"] = 2
    dump(args["provenance"], meta)
    report = validate_data(**args)
    assert {"provenance_sha256", "provenance"} <= codes(report)


def test_crop_metadata_counts_and_unknown_classes(dataset):
    args, _, _ = dataset
    stats = json.loads(args["crop_meta"].read_text())
    stats["kept_classes"]["can"] = 1
    dump(args["crop_meta"], stats)
    extra = args["crops_dir"] / "unknown"
    extra.mkdir()
    Image.new("RGB", (8, 8)).save(extra / "x.jpg")
    assert {"crop_counts", "metadata", "crop_class"} <= codes(validate_data(**args))


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("embeddings", np.full((4, 3), np.nan), "cache_finite"),
        ("embeddings", np.full((4, 3), np.inf), "cache_finite"),
        ("embeddings", np.ones((4, 2), dtype=np.float32), "cache_dimension"),
        ("embeddings", np.ones(4, dtype=np.float32), "cache_array"),
        ("embeddings", np.empty((0, 3), dtype=np.float32), "cache_array"),
        ("embeddings", np.ones((4, 3), dtype=np.int64), "cache_array"),
        ("labels", np.array(["can"] * 4, dtype=object), "cache_array"),
        ("labels", np.array(["can"]), "cache_rows"),
        ("labels", np.array(["can", "can", "unknown", "bottle"]), "cache_label"),
        ("labels", np.array(["bottle", "can", "can", "bottle"]), "cache_label"),
        ("crop_paths", np.array(["../outside.jpg"] * 4), "cache_path"),
    ],
)
def test_cache_types_dimensions_finiteness_and_paths(dataset, key, value, code):
    args, _, arrays = dataset
    arrays[key] = value
    np.savez_compressed(args["cache"], **arrays)
    report = validate_data(**args)
    assert not report["ok"] and code in codes(report), report


def test_cache_missing_extra_and_duplicate_members(dataset):
    args, _, arrays = dataset
    np.savez_compressed(args["cache"], labels=arrays["labels"], extra=np.zeros(1))
    assert "cache_members" in codes(validate_data(**args))
    np.savez_compressed(args["cache"], **arrays)
    with zipfile.ZipFile(args["cache"], "a") as z:
        payload = z.read("labels.npy")
        with pytest.warns(UserWarning):
            z.writestr("labels.npy", payload)
    assert "cache_members" in codes(validate_data(**args))


def test_cache_missing_duplicate_paths_and_fingerprint(dataset):
    args, _, arrays = dataset
    arrays["crop_paths"][1] = arrays["crop_paths"][0]
    np.savez_compressed(args["cache"], **arrays)
    assert {"duplicate_path", "cache_paths"} <= codes(validate_data(**args))
    (args["crops_dir"] / "bottle/b.jpg").unlink()
    assert {"fingerprint", "cache_path"} <= codes(validate_data(**args))


def test_bounded_npz_streaming_checks_late_values_and_fortran_order(dataset, monkeypatch):
    args, _, arrays = dataset
    # A small chunk limit proves every block is consumed without large fixtures.
    monkeypatch.setattr("src.quobo.data_validation._CHUNK_BYTES", 16)
    arrays["embeddings"][-1, -1] = np.nan
    np.savez_compressed(args["cache"], **arrays)

    def forbidden(*a, **kw):
        raise AssertionError("validator must not np.load the full cache")

    monkeypatch.setattr(np, "load", forbidden)
    assert "cache_finite" in codes(validate_data(**args))
    report = validate_data(**args, max_cache_bytes=1)
    assert "cache_size" in codes(report) and not report["ok"]


def test_truncated_npz_is_invalid_not_exception(dataset):
    args, _, _ = dataset
    args["cache"].write_bytes(args["cache"].read_bytes()[:50])
    assert "cache" in codes(validate_data(**args))


def test_relative_cache_paths_use_explicit_base(dataset):
    args, _, arrays = dataset
    root = args["crops_dir"].parent
    arrays["crop_paths"] = np.array([str(Path(p).relative_to(root)) for p in arrays["crop_paths"]])
    np.savez_compressed(args["cache"], **arrays)
    assert validate_data(**args, cache_path_base=root)["ok"]


def test_symlink_containment_when_supported(dataset, tmp_path):
    args, coco, _ = dataset
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (8, 8)).save(outside)
    link = args["images_dir"] / "linked.jpg"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks requires platform privileges")
    coco["images"][0]["file_name"] = "linked.jpg"
    dump(args["annotations"], coco)
    assert "path" in codes(
        validate_data(annotations=args["annotations"], images_dir=args["images_dir"])
    )


def test_no_vacuous_success_or_missing_companions(tmp_path, dataset):
    args, _, _ = dataset
    assert not validate_data()["ok"]
    assert not validate_data(images_dir=tmp_path / "missing")["ok"]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "empty_images" in codes(validate_data(images_dir=empty))
    assert "arguments" in codes(validate_data(cache=args["cache"]))
    assert "arguments" in codes(validate_data(crops_dir=args["crops_dir"]))
    assert "arguments" in codes(validate_data(provenance=args["provenance"]))
    assert "arguments" in codes(validate_data(crop_meta=args["crop_meta"]))
    assert "arguments" in codes(validate_data(cache_meta=args["cache_meta"]))
    assert "arguments" in codes(validate_data(**{**args, "classes": ["../bad"]}))


def test_cli_help_is_inert_and_invalid_input_returns_json(tmp_path, dataset):
    args, _, _ = dataset
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    before_help = {
        p.relative_to(tmp_path): (p.stat().st_mtime_ns, p.read_bytes())
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    help_run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_data.py"), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert help_run.returncode == 0 and "--provenance" in help_run.stdout
    after_help = {
        p.relative_to(tmp_path): (p.stat().st_mtime_ns, p.read_bytes())
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert help_run.stderr == "" and before_help == after_help
    invalid = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_data.py"),
            "--annotations",
            str(tmp_path / "absent.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert invalid.returncode == 1 and not json.loads(invalid.stdout)["ok"]
    no_args = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_data.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert no_args.returncode == 1 and "arguments" in codes(json.loads(no_args.stdout))
    valid = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_data.py"),
            "--annotations",
            str(args["annotations"]),
            "--images-dir",
            str(args["images_dir"]),
            "--provenance",
            str(args["provenance"]),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert valid.returncode == 0 and json.loads(valid.stdout)["ok"]
