"""Stage 1: TACO dataset — download annotations/images, crop annotated objects,
build a classification dataset of per-class crop folders.

TACO (Proenca & Simoes, arXiv:2003.06975) ships COCO-style annotations.
We crop each annotation bbox (with 10% margin) and save as
data/processed/crops/<class>/<img_id>_<ann_id>.jpg
"""
import json
import logging
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import requests
from tqdm import tqdm

from .config import ROOT

log = logging.getLogger(__name__)

TACO_URL = "https://github.com/pedropro/TACO/raw/master/data/annotations.json"
SUPERCAT_MAP_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/classes.csv"
)


MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # annotations.json is ~10 MB; anything bigger is wrong


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("already downloaded: %s", dest.name)
        return dest
    resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    written = 0
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            written += len(chunk)
            if written > MAX_DOWNLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise ValueError(f"download exceeded {MAX_DOWNLOAD_BYTES} bytes: {url}")
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def download_taco_images(ann_path: Path, images_dir: Path) -> None:
    """Download images from Flickr 640px URLs (same as official download.py).
    Resumable: skips files that already exist. Broken links are tolerated —
    TACO's own downloader has the same behavior."""
    import io

    from PIL import Image

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    images_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for im in coco["images"]:
        p = images_dir / im["file_name"]
        if not p.exists():
            todo.append((im, p))
    log.info("images to fetch: %d / %d", len(todo), len(coco["images"]))

    session = requests.Session()
    failed = []
    for im, p in tqdm(todo, desc="flickr"):
        # flickr_640_url is a fast ~90KB resize; flickr_url / olm-s3 entries are
        # multi-MB originals served at ~40KB/s — not worth the wait for 224px crops
        url = im.get("flickr_640_url")
        if not url:
            failed.append((im["file_name"], "no-640-url"))
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content))
            img.save(p)
        except Exception as e:  # noqa: BLE001 — dead Flickr links are expected
            failed.append((im["file_name"], str(e)[:60]))
    if failed:
        log.warning("%d images failed (dead Flickr links):", len(failed))
        for fn, err in failed[:10]:
            log.warning("  %s: %s", fn, err)


def download_taco(raw_dir: Path) -> tuple[Path, Path]:
    """Download annotations + images. Returns (annotations_path, images_dir)."""
    ann_path = raw_dir / "annotations.json"
    if not ann_path.exists():
        download_file(TACO_URL, ann_path, "annotations.json")

    images_dir = raw_dir / "images"
    download_taco_images(ann_path, images_dir)
    return ann_path, images_dir


def load_supercat_map(cfg_classes: list[str]) -> dict[str, str]:
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

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    leaf_map = load_supercat_map(cfg_classes)
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
            x, y, w, h = (c * s for c, s in zip(ann["bbox"], (sx, sy, sx, sy)))
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
            crop_name = f"{rec['file_name'].replace('/', '_').replace('.jpg', '')}_ann{ann['id']}.jpg"
            dest = cls_dir / crop_name
            if not dest.resolve().is_relative_to(out_dir.resolve()):
                skipped["path_escape"] += 1
                continue
            cv2.imwrite(str(dest), img[y0:y1, x0:x1])
            per_class[coarse] += 1

    # drop classes below the minimum count
    removed = {}
    for cls in list(cfg_classes):
        if per_class[cls] < min_images_per_class:
            d = out_dir / cls
            if d.exists():
                shutil.rmtree(d)
            removed[cls] = per_class[cls]

    stats = {
        "per_class": dict(per_class),
        "removed_classes": removed,
        "skipped": dict(skipped),
        "kept_classes": {c: per_class[c] for c in cfg_classes if per_class[c] >= min_images_per_class},
    }
    return stats


def run_data_prep(cfg: dict) -> dict:
    raw = ROOT / "data" / "raw"
    crops_dir = ROOT / "data" / "processed" / "crops"
    ann_path, images_dir = download_taco(raw)
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
    from .config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml")
    s = run_data_prep(cfg)
    print("per-class counts:", s["per_class"])
    print("removed (below min):", s["removed_classes"])
