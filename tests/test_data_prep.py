"""Unit test for data_prep: bounding-box clamp in actual image space (BBOX-001)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_bbox_clamp_actual_image_dims(tmp_path):
    import cv2

    from src.quobo.data_prep import build_crops

    imgs = tmp_path / "images"
    (imgs / "batch_1").mkdir(parents=True)
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[:, 200:] = 255
    cv2.imwrite(str(imgs / "batch_1/000001.jpg"), img)
    coco = {
        "categories": [{"id": 1, "name": "Clear plastic bottle"}],
        "images": [{"id": 0, "width": 200, "height": 100, "file_name": "batch_1/000001.jpg"}],
        "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [150, 25, 40, 20]}],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    stats = build_crops(ann, imgs, tmp_path / "crops", ["bottle"], 1)
    crop = cv2.imread(str(next((tmp_path / "crops" / "bottle").glob("*.jpg"))))
    # bbox scaled 2x: x=300,y=50,w=80,h=40; 10% margin; clamp to 400x200
    assert crop.shape[:2] == (48, 96)
    assert stats["per_class"]["bottle"] == 1
