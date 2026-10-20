"""Stage 1: TACO dataset — download annotations/images, crop annotated objects,
build a classification dataset of per-class crop folders.

TACO (Proenca & Simoes, arXiv:2003.06975) ships COCO-style annotations.
We crop each annotation bbox (with 10% margin) and save as
data/processed/crops/<class>/<img_id>_<ann_id>.jpg
"""

import json
import logging
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import requests
from tqdm import tqdm

from .config import ROOT

log = logging.getLogger(__name__)

TACO_URL = "https://github.com/pedropro/TACO/raw/master/data/annotations.json"


MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # annotations.json is ~10 MB; anything bigger is wrong
MAX_IMG_BYTES = 20 * 1024 * 1024  # flickr_640 images are ~90 KB; cap hostile redirects
# Decoded-pixel budget. The byte cap only bounds the COMPRESSED size: a small
# PNG can decode to a huge bitmap (518 KB -> 13000x13000 = 507 MB). TACO
# originals are at most a few MP, so 40 MP is far above any legitimate image
# and far below a decompression-bomb footprint.
MAX_IMAGE_PIXELS = 40_000_000
# Retries per image for transient failures (timeout, connection reset, 5xx).
# 404/410 are permanent and are not retried.
IMG_RETRIES = 3
# Allowed image hosts. TACO serves images from Flickr's CDN (*.staticflickr.com)
# and the OLM S3 bucket (olm-s3.s3.eu-west-1.amazonaws.com) — verified against
# the shipped annotations.json. The annotations file is treated as untrusted
# input (see _validate_dataset_paths), so its URLs are allowlisted too:
# without this, a crafted file_name/URL could point the fetcher at a cloud
# metadata endpoint (169.254.169.254) or an internal service.
ALLOWED_IMAGE_HOSTS = ("staticflickr.com", "olm-s3.s3.eu-west-1.amazonaws.com")


def _validate_image_url(url: str, context: str = "image") -> str:
    """Reject a URL that is not https on an allowlisted host.

    Returns the URL unchanged when acceptable; raises ValueError otherwise.
    Subdomains of the allowed hosts are accepted (flickr uses farm1..farm66).

    A string with no scheme is left alone: it cannot address a network host,
    so it carries no SSRF risk — the request layer rejects it on its own."""
    from urllib.parse import urlparse

    if not url:
        raise ValueError(f"empty {context} URL")
    parsed = urlparse(url)
    if not parsed.scheme:
        # Not an absolute URL — not an SSRF vector. Leave it to the transport.
        return url
    if parsed.scheme != "https":
        raise ValueError(f"{context} URL must use https, got {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_IMAGE_HOSTS):
        raise ValueError(f"{context} URL host {host!r} is not allowlisted")
    return url


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Stream a URL to ``dest`` atomically.

    The body is written to a sibling temp file and published with
    ``os.replace`` only after a clean finish. A partial download must never
    occupy ``dest``: the "already downloaded" check below treats existence as
    proof of a complete fetch, so a truncated file left behind by a mid-stream
    failure would be accepted forever and permanently corrupt the dataset
    (for annotations.json, the whole pipeline).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("already downloaded: %s", dest.name)
        return dest
    resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    written = 0
    temp = dest.with_name(dest.name + ".part")
    try:
        with (
            open(temp, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=1 << 20):
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"download exceeded {MAX_DOWNLOAD_BYTES} bytes: {url}")
                f.write(chunk)
                bar.update(len(chunk))
        os.replace(temp, dest)
    finally:
        # Any failure (cap exceeded, connection reset, timeout, disk error)
        # leaves dest untouched and removes the partial file.
        temp.unlink(missing_ok=True)
    return dest


def _validate_dataset_paths(coco: dict, root: Path, context: str) -> None:
    """Reject crafted file_name entries before any decode/download.

    COCO file_name values are treated as dataset-relative POSIX-style names.
    Any absolute path, drive letter, '..' segment, or traversal that would
    escape *root* is a data-integrity error, not a per-image skip: silently
    skipping can hide a corrupted or tampered snapshot.
    """
    for im in coco.get("images", []):
        fn = im.get("file_name", "")
        if not fn or PurePosixPath(fn).is_absolute() or PureWindowsPath(fn).is_absolute():
            raise ValueError(f"unsafe {context} filename: {fn!r}")
        if ".." in PurePosixPath(fn).parts or ".." in PureWindowsPath(fn).parts:
            raise ValueError(f"unsafe {context} filename: {fn!r}")
        if not (root / fn).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unsafe {context} filename escapes root: {fn!r}")


def download_taco_images(ann_path: Path, images_dir: Path) -> None:
    """Download images from Flickr 640px URLs (same as official download.py).
    Resumable: skips files that already exist. Broken links are tolerated —
    TACO's own downloader has the same behavior."""
    import io

    from PIL import Image

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    _validate_dataset_paths(coco, images_dir, "downloader")

    images_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for im in coco["images"]:
        p = images_dir / im["file_name"]
        if not p.exists():
            todo.append((im, p))
    log.info("images to fetch: %d / %d", len(todo), len(coco["images"]))

    session = requests.Session()
    failed = []
    dead_links = 0
    transient = 0
    for im, p in tqdm(todo, desc="flickr"):
        # flickr_640_url is a fast ~90KB resize; flickr_url / olm-s3 entries are
        # multi-MB originals served at ~40KB/s — not worth the wait for 224px crops
        url = im.get("flickr_640_url")
        if not url:
            failed.append((im["file_name"], "no-640-url"))
            continue
        try:
            url = _validate_image_url(url, "flickr_640_url")
        except ValueError as exc:
            failed.append((im["file_name"], str(exc)[:60]))
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        # Retry transient failures (timeout, reset, 5xx) with backoff. A
        # single 30 s timeout on one of 1,500 images would otherwise drop
        # that image from the run's crop counts, and the pipeline would
        # continue over a silently degraded dataset. 404/410 are permanent
        # and are not retried.
        last_err = ""
        for attempt in range(IMG_RETRIES):
            try:
                with session.get(url, timeout=30, stream=True) as r:
                    r.raise_for_status()
                    received = 0
                    payload = io.BytesIO()
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        received += len(chunk)
                        if received > MAX_IMG_BYTES:
                            raise ValueError(f"image exceeds {MAX_IMG_BYTES} byte cap")
                        payload.write(chunk)
                    img = Image.open(payload)
                    # Pixel guard: the byte cap only bounds the COMPRESSED size.
                    # A small PNG can decode to a huge bitmap (e.g. 518 KB ->
                    # 13000x13000 = 507 MB), so bound the decoded dimensions too.
                    if img.width * img.height > MAX_IMAGE_PIXELS:
                        raise ValueError(
                            f"image {img.width}x{img.height} exceeds the "
                            f"{MAX_IMAGE_PIXELS} pixel budget"
                        )
                    img.load()
                    # Publish atomically: a half-written image at the final path
                    # would be treated as downloaded on the next run (p.exists())
                    # and silently dropped from the dataset forever.
                    temp = p.with_name(p.name + ".part")
                    try:
                        # format= is required: the .part suffix is not a
                        # format PIL can infer from.
                        img.save(temp, format="JPEG")
                        os.replace(temp, p)
                    finally:
                        temp.unlink(missing_ok=True)
                last_err = ""
                break
            except Exception as e:  # noqa: BLE001 — dead Flickr links are expected
                last_err = f"{type(e).__name__}: {e}"[:60]
                if isinstance(e, ValueError) or "404" in last_err or "410" in last_err:
                    break  # permanent: size cap, or the photo is gone
                if attempt < IMG_RETRIES - 1:
                    time.sleep(2 * (attempt + 1))
        if last_err:
            failed.append((im["file_name"], last_err))
            if "404" in last_err or "410" in last_err:
                dead_links += 1
            else:
                transient += 1
    if failed:
        log.warning("%d images failed (dead Flickr links):", len(failed))
        for fn, err in failed[:10]:
            log.warning("  %s: %s", fn, err)
    if transient or dead_links:
        # Distinguish a permanently removed photo from a transient failure —
        # a degraded dataset must not pass silently.
        log.warning(
            "image download outcome: %d permanently dead (404/410, not retried), "
            "%d still failing after %d attempts. Those images are absent from "
            "the dataset; re-run to recover the transient ones.",
            dead_links,
            transient,
            IMG_RETRIES,
        )


def record_provenance(ann_path: Path, raw_dir: Path) -> dict:
    """OPEN-6: pin the dataset snapshot — source URL, sha256, mtime, counts.
    The survey found TACO v1.0 (1,500 images) vs the living GitHub dump
    (3,831) diverge; the paper must state which snapshot produced the results."""
    import hashlib
    from datetime import datetime, timezone

    h = hashlib.sha256(ann_path.read_bytes()).hexdigest()
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    prov = {
        "source_url": TACO_URL,
        "sha256": h,
        "file_mtime": datetime.fromtimestamp(ann_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "n_images": len(coco["images"]),
        "n_annotations": len(coco["annotations"]),
        "note": "snapshot recorded at first pipeline run; v1.0 paper release is 1500 images",
    }
    with open(raw_dir / "annotations_provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    log.info("provenance recorded: sha256=%s... n_images=%d", h[:12], prov["n_images"])
    return prov


def download_taco(raw_dir: Path) -> tuple[Path, Path]:
    """Download annotations + images. Returns (annotations_path, images_dir)."""
    ann_path = raw_dir / "annotations.json"
    if not ann_path.exists():
        download_file(TACO_URL, ann_path, "annotations.json")

    images_dir = raw_dir / "images"
    download_taco_images(ann_path, images_dir)
    return ann_path, images_dir


def load_supercat_map() -> dict[str, str]:
    """Map each of TACO's 60 leaf categories to one of the configured coarse classes.

    Uses the paper's own grouping: leaf categories whose supercategory matches a
    configured class name map directly; everything else falls into 'other'.
    """
    leaf_to_super = {}
    with open(ROOT / "configs" / "taco_taxonomy.csv", encoding="utf-8") as f:
        import csv

        for row in csv.DictReader(f):
            leaf_to_super[row["leaf"]] = row["coarse"]
    return leaf_to_super


def build_crops(
    ann_path: Path,
    images_dir: Path,
    out_dir: Path,
    cfg_classes: list[str],
    min_images_per_class: int,
) -> dict:
    """Crop annotated objects into per-class folders. Returns a stats dict."""
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    _validate_dataset_paths(coco, images_dir, "crop source")

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    leaf_map = load_supercat_map()
    img_id_to_rec = {im["id"]: im for im in coco["images"]}

    per_class = Counter()
    skipped = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # group annotations by image so each image decodes once (worst image has
    # 90 annotations — naive per-annotation decode re-reads it 90 times)
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        leaf = cat_id_to_name[ann["category_id"]]
        coarse = leaf_map.get(leaf, "Unlabeled litter")
        if coarse in cfg_classes:
            anns_by_img[ann["image_id"]].append((ann, coarse))

    for img_id, ann_list in anns_by_img.items():
        rec = img_id_to_rec[img_id]
        img_path = images_dir / rec["file_name"]
        if not img_path.exists():
            skipped["missing_image"] += len(ann_list)
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped["unreadable"] += len(ann_list)
            continue
        # downloaded images may be resized (640px Flickr variants) — annotation
        # coords live in the ORIGINAL image space, so rescale the bbox
        H, W = img.shape[:2]
        sx = W / rec["width"]
        sy = H / rec["height"]
        for ann, coarse in ann_list:
            x, y, w, h = (c * s for c, s in zip(ann["bbox"], (sx, sy, sx, sy), strict=True))
            m = 0.10  # margin
            x0 = max(0, int(x - m * w))
            y0 = max(0, int(y - m * h))
            x1 = min(W, int(x + w * (1 + m)))  # clamp to the ACTUAL image dims
            y1 = min(H, int(y + h * (1 + m)))
            if x1 - x0 < 8 or y1 - y0 < 8:
                skipped["too_small"] += 1
                continue
            cls_dir = out_dir / coarse
            cls_dir.mkdir(exist_ok=True)
            # dataset filenames are semi-trusted: verify the destination stays
            # inside the crops root (blocks path traversal via crafted file_name)
            # Both separators are flattened: a Windows-style "subdir\img.jpg"
            # would otherwise create a nested path that does not exist, making
            # cv2.imwrite fail silently and leaving the crop count inflated.
            flat_name = rec["file_name"].replace("/", "_").replace("\\", "_")
            crop_name = f"{flat_name.replace('.jpg', '')}_ann{ann['id']}.jpg"
            dest = cls_dir / crop_name
            if not dest.resolve().is_relative_to(out_dir.resolve()):
                skipped["path_escape"] += 1
                continue
            if not cv2.imwrite(str(dest), img[y0:y1, x0:x1]):
                # imwrite returns False on failure (unwritable path, bad codec)
                # and writes nothing — never count a crop that was not saved.
                skipped["write_failed"] += 1
                continue
            per_class[coarse] += 1

    # drop classes below the minimum count
    removed = {}
    out_root = out_dir.resolve()
    for cls in list(cfg_classes):
        if per_class[cls] < min_images_per_class:
            d = out_dir / cls
            # A configured class name becomes a directory that is rmtree'd —
            # a class named ".." would resolve to the PARENT of the crops root
            # and delete it. Refuse anything that is not a plain child name.
            if d.resolve() != out_root and not d.resolve().is_relative_to(out_root):
                raise ValueError(f"unsafe class name for directory removal: {cls!r}")
            if d.exists():
                shutil.rmtree(d)
            removed[cls] = per_class[cls]

    stats = {
        "per_class": dict(per_class),
        "removed_classes": removed,
        "skipped": dict(skipped),
        "kept_classes": {
            c: per_class[c] for c in cfg_classes if per_class[c] >= min_images_per_class
        },
    }
    return stats


def run_data_prep(cfg: dict) -> dict:
    raw = ROOT / "data" / "raw"
    crops_dir = ROOT / "data" / "processed" / "crops"
    ann_path, images_dir = download_taco(raw)
    record_provenance(ann_path, raw)
    stats = build_crops(
        ann_path,
        images_dir,
        crops_dir,
        cfg["data"]["classes"],
        cfg["data"]["min_images_per_class"],
    )
    with open(ROOT / "data" / "processed" / "data_prep_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import load_config, validate_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment_6class.yaml")
    validate_config(cfg)  # enforce range/cross-field rules, not just key names
    s = run_data_prep(cfg)
    print("per-class counts:", s["per_class"])
    print("removed (below min):", s["removed_classes"])
