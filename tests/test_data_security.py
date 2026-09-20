"""Dataset paths must be rejected before reading or writing outside the root."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quobo import data_prep


@pytest.mark.parametrize(
    "filename", ["../outside.jpg", "..\\outside.jpg", "/outside.jpg", "C:\\outside.jpg"]
)
def test_build_crops_rejects_unsafe_source_before_decode(tmp_path, monkeypatch, filename):
    images = tmp_path / "images"
    images.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"not a real image")
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": filename, "width": 16, "height": 16}],
                "categories": [{"id": 1, "name": "Bottle"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 16, 16]}],
            }
        )
    )
    monkeypatch.setattr(data_prep, "load_supercat_map", lambda _: {"Bottle": "bottle"})
    decoded = []

    def decode(path):
        decoded.append(path)
        return np.zeros((16, 16, 3), dtype=np.uint8)

    monkeypatch.setattr(data_prep.cv2, "imread", decode)
    with pytest.raises(ValueError, match="path|filename"):
        data_prep.build_crops(ann_path, images, tmp_path / "crops", ["bottle"], 1)
    assert decoded == []


def test_downloader_rejects_unsafe_paths_before_network(tmp_path, monkeypatch):
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {"images": [{"file_name": "../escape.jpg", "flickr_640_url": "https://invalid.test/x"}]}
        )
    )

    def forbidden_session():
        pytest.fail("Network client constructed before path validation")

    monkeypatch.setattr(data_prep.requests, "Session", forbidden_session)
    with pytest.raises(ValueError, match="path|filename"):
        data_prep.download_taco_images(ann_path, tmp_path / "images")


def test_fetch_missing_rejects_unsafe_paths_before_network(tmp_path, monkeypatch):
    """Q08 regression: fetch_missing validates every file_name before any
    network client is constructed or any destination directory is created."""
    import scripts.fetch_missing as FM

    ann = tmp_path / "annotations.json"
    ann.write_text(
        json.dumps(
            {
                "images": [
                    {"file_name": r"..\escape.jpg", "flickr_640_url": "https://invalid.test/x"}
                ]
            }
        )
    )

    def forbidden_session():
        pytest.fail("Network client constructed before path validation")

    monkeypatch.setattr(FM.requests, "Session", forbidden_session)
    monkeypatch.setattr(FM, "ANN", ann)
    monkeypatch.setattr(FM, "IMAGES", tmp_path / "images")
    with pytest.raises(ValueError, match="path|filename"):
        FM.main()
    assert not (tmp_path / "escape.jpg").exists()


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        return iter(self._chunks)


def test_fetch_one_enforces_byte_cap_streaming(tmp_path, monkeypatch):
    """Q09 regression: an oversized response is aborted mid-stream by the byte
    cap — the body is never fully buffered — and no destination is written."""
    import scripts.fetch_missing as FM

    dest = tmp_path / "img.jpg"
    dest.write_bytes(b"previous-good-image")
    seen = []

    class Session:
        def get(self, url, timeout=None, stream=False):
            seen.append(stream)
            return _FakeResponse([b"x" * (1 << 16)] * 400)  # ~25.6 MB total

    monkeypatch.setattr(FM.requests, "Session", Session)
    monkeypatch.setattr(FM.time, "sleep", lambda _s: None)
    status, err = FM.fetch_one({"flickr_640_url": "u"}, dest)
    assert status == "failed"
    assert "exceeds" in err
    assert dest.read_bytes() == b"previous-good-image"
    assert not [p for p in tmp_path.iterdir() if p.name != "img.jpg" and p.suffix == ".jpg"]


def test_fetch_one_atomic_publish_and_retry(tmp_path, monkeypatch):
    """A transient failure leaves no temp litter; a valid small response is
    published atomically at the destination and downscaled to MAX_DIM."""
    import io as _io

    from PIL import Image as PILImage

    import scripts.fetch_missing as FM

    dest = tmp_path / "img.jpg"
    big = PILImage.new("RGB", (2000, 1000), color=(10, 20, 30))
    buf = _io.BytesIO()
    big.save(buf, format="JPEG", quality=95)
    payload = buf.getvalue()
    assert len(payload) < FM.MAX_IMG_BYTES

    attempts = {"n": 0}

    class FlakySession:
        def get(self, url, timeout=None, stream=False):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("simulated transient failure")
            return _FakeResponse([payload])

    monkeypatch.setattr(FM.requests, "Session", FlakySession)
    monkeypatch.setattr(FM.time, "sleep", lambda _s: None)
    status, err = FM.fetch_one({"flickr_640_url": "u"}, dest)
    assert status == "ok", err
    assert attempts["n"] == 2
    with PILImage.open(dest) as saved:
        assert max(saved.size) <= FM.MAX_DIM
    leftovers = [p for p in tmp_path.iterdir() if p.name != dest.name]
    assert leftovers == [], f"temp litter: {leftovers}"
