"""Fetch TACO images that lack a flickr_640_url using the olm-s3 fallback,
with retries + threading + on-save downscale. Resumable: skips existing files.

Usage: python scripts/fetch_missing.py
"""
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "raw" / "annotations.json"
IMAGES = ROOT / "data" / "raw" / "images"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fetch_missing")

MAX_DIM = 1024
MAX_IMG_BYTES = 20 * 1024 * 1024  # ~14 MB is the largest legitimate olm-s3 original
WORKERS = 8
RETRIES = 4


def fetch_one(session: requests.Session, im: dict, dest: Path) -> tuple[str, str]:
    urls = [u for u in (im.get("flickr_640_url"), im.get("flickr_url")) if u]
    last_err = "no-url"
    for attempt in range(RETRIES):
        for url in urls:
            try:
                r = session.get(url, timeout=(10, 120))
                r.raise_for_status()
                if len(r.content) > MAX_IMG_BYTES:
                    raise ValueError(f"image response {len(r.content)} bytes exceeds cap")
                img = Image.open(io.BytesIO(r.content))
                img = img.convert("RGB")
                if max(img.size) > MAX_DIM:
                    img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
                dest.parent.mkdir(parents=True, exist_ok=True)
                img.save(dest, quality=90)
                return "ok", ""
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"[:80]
                if isinstance(e, (ValueError,)) or "404" in str(last_err) or "410" in str(last_err):
                    # size violation or permanently dead link — no point retrying
                    return "failed", last_err
        time.sleep(2 * (attempt + 1))
    return "failed", last_err


def main() -> int:
    coco = json.loads(ANN.read_text(encoding="utf-8"))
    todo = [im for im in coco["images"] if not (IMAGES / im["file_name"]).exists()]
    log.info("missing images: %d / %d", len(todo), len(coco["images"]))
    if not todo:
        return 0

    session = requests.Session()
    ok = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(fetch_one, session, im, IMAGES / im["file_name"]): im["file_name"]
            for im in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            status, err = fut.result()
            if status == "ok":
                ok += 1
            else:
                failed += 1
                log.warning("FAILED %s: %s", futs[fut], err)
            if i % 25 == 0:
                rate = i / (time.time() - t0)
                log.info("%d/%d done (ok=%d failed=%d, %.1f img/min)", i, len(todo), ok, failed, rate * 60)
    log.info("FINISHED ok=%d failed=%d elapsed=%.1f min", ok, failed, (time.time() - t0) / 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
