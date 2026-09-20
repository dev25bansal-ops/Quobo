"""Unit tests for the checkpointed embedding extraction and atomic caching
(Enhancement 2: crash-resume + publish-failure recovery)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_extract_embeddings_checkpoint_resume(tmp_path, monkeypatch):
    """extract_embeddings with a checkpoint dir resumes: a second call with a
    pre-existing partial checkpoint skips already-embedded rows."""
    import cv2

    from src.quobo import features as F

    # two tiny valid crops (load_img just needs decodable images)
    crops = tmp_path / "crops"
    for cls in ("a", "b"):
        d = crops / cls
        d.mkdir(parents=True)
        for i in range(3):
            img = np.full((64, 64, 3), 30 + i * 10, dtype=np.uint8)
            cv2.imwrite(str(d / f"{i}.jpg"), img)

    # fake model: probe + predict return a fixed 8-d vector per sample;
    # count samples predicted so we can prove the resume path skips work
    calls = {"samples": 0}

    class FakeModel:
        def predict(self, imgs, verbose=0):
            calls["samples"] += imgs.shape[0]
            return np.ones((imgs.shape[0], 8), dtype=np.float32)

    fake = FakeModel()
    monkeypatch.setattr(
        F.tf.keras.utils,
        "load_img",
        lambda p, target_size=(224, 224): np.full((64, 64, 3), 20, dtype=np.uint8),
    )

    ckpt_dir = tmp_path / "ckpt"
    df1 = F.extract_embeddings(fake, crops, ["a", "b"], checkpoint_dir=ckpt_dir, checkpoint_tag="t")
    assert df1.shape[0] == 6 and df1.shape[1] == 8 + 2  # 8 emb + label + crop_path
    first_samples = calls["samples"]  # 1 (probe) + 6 (all rows)

    # simulate a crash after embedding 3 rows: write a partial checkpoint
    # marking 3 done; the resumed call should only predict the 3 todo rows
    import numpy as _np

    paths = list(df1["crop_path"])
    partial = ckpt_dir / "t.partial.npz"
    _np.savez_compressed(
        partial,
        paths=_np.array(paths),
        done=_np.array([True, True, True, False, False, False]),
        embs=_np.ones((6, 8), dtype="float32"),
        fingerprint=_np.array(
            json.dumps(F.crops_content_fingerprint(crops, ["a", "b"]), sort_keys=True)
        ),
    )
    calls["samples"] = 0
    df2 = F.extract_embeddings(fake, crops, ["a", "b"], checkpoint_dir=ckpt_dir, checkpoint_tag="t")
    # resumed: probe(1) + 3 todo rows = 4 samples < first call's 7
    assert (
        calls["samples"] < first_samples
    ), f"expected resume to skip work ({calls['samples']} < {first_samples})"
    assert df2.shape == df1.shape


@pytest.mark.parametrize("stale", [False, True])
def test_raw_wrapper_resumes_sparse_checkpoint(tmp_path, monkeypatch, stale):
    from src.quobo import features as F

    monkeypatch.setattr(F, "ROOT", tmp_path)
    crops = tmp_path / "crops" / "a"
    crops.mkdir(parents=True)
    paths = []
    for i in range(4):
        path = crops / f"{i}.jpg"
        path.write_bytes(b"fixture")
        paths.append(str(path))
    ckpt = tmp_path / "data/features/_checkpoints/raw_a.partial.npz"
    fp = F.crops_content_fingerprint(crops.parent, ["a"])
    if stale:
        fp["files_hash"] = "old"
    F._atomic_savez(
        ckpt,
        paths=np.array(paths),
        done=np.array([False, True, False, True]),
        embs=np.full((4, 2), 7.0, dtype=np.float32),
        fingerprint=np.array(json.dumps(fp, sort_keys=True)),
    )
    calls = []

    class Model:
        def predict(self, imgs, verbose=0):
            calls.append(len(imgs))
            return np.ones((len(imgs), 2), dtype=np.float32)

    monkeypatch.setattr(F, "load_model", Model)
    monkeypatch.setattr(
        F.tf.keras.utils, "load_img", lambda *a, **kw: np.zeros((8, 8, 3), dtype=np.float32)
    )
    cfg = {"data": {"classes": ["a"], "crops_dir_override": "crops"}}
    result = F.run_raw_embeddings(cfg)
    assert calls == [1, 4 if stale else 2]
    assert result["e0"].tolist() == ([1, 1, 1, 1] if stale else [1, 7, 1, 7])
    assert not ckpt.exists()
    calls.clear()
    pd.testing.assert_frame_equal(F.run_raw_embeddings(cfg), result)
    assert calls == []


def test_atomic_checkpoint_failure_preserves_previous_archive(tmp_path, monkeypatch):
    from src.quobo import features as F

    path = tmp_path / "checkpoint.npz"
    F._atomic_savez(path, data=np.array([42]))
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted compression")

    monkeypatch.setattr(F.np, "savez_compressed", fail)
    with pytest.raises(OSError, match="interrupted compression"):
        F._atomic_savez(path, data=np.array([99]))
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_raw_wrapper_recovers_after_cache_publish_failure(tmp_path, monkeypatch):
    from src.quobo import features as F

    monkeypatch.setattr(F, "ROOT", tmp_path)
    crops = tmp_path / "crops/a"
    crops.mkdir(parents=True)
    (crops / "0.jpg").write_bytes(b"fixture")
    calls = []

    class Model:
        def predict(self, imgs, verbose=0):
            calls.append(len(imgs))
            return np.full((len(imgs), 2), 3.0, dtype=np.float32)

    monkeypatch.setattr(F, "load_model", Model)
    monkeypatch.setattr(
        F.tf.keras.utils, "load_img", lambda *a, **kw: np.zeros((8, 8, 3), dtype=np.float32)
    )
    save = F._atomic_savez

    def fail_final(path, **arrays):
        if path.name == "raw_a.npz":
            raise OSError("simulated final publication failure")
        save(path, **arrays)

    monkeypatch.setattr(F, "_atomic_savez", fail_final)
    cfg = {"data": {"classes": ["a"], "crops_dir_override": "crops"}}
    with pytest.raises(OSError, match="final publication"):
        F.run_raw_embeddings(cfg)
    checkpoint = tmp_path / "data/features/_checkpoints/raw_a.partial.npz"
    assert checkpoint.exists()
    monkeypatch.setattr(F, "_atomic_savez", save)
    calls.clear()
    result = F.run_raw_embeddings(cfg)
    assert calls == [1]  # output-width probe only; completed rows are reused
    assert result["e0"].tolist() == [3.0]
    assert not checkpoint.exists()
