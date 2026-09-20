"""Unit test: ONNX export parity on a tiny synthetic model (Enhancement 8)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_onnx_pipeline_parity(tmp_path):
    """The exporter's core pattern (StandardScaler+PCA -> RBF-SVM -> ONNX)
    must reproduce sklearn predictions. Uses a tiny synthetic model so CI
    doesn't need the real MobileNetV2 embeddings."""
    ort = pytest.importorskip("onnxruntime", reason="edge extra not installed")
    pytest.importorskip("skl2onnx", reason="edge extra not installed")
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 20))
    y = (X[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=8, random_state=0).fit(scaler.transform(X))
    X50 = pca.transform(scaler.transform(X))
    sel = [0, 1, 3, 4, 5, 6, 7]
    rbf = SVC(kernel="rbf", random_state=0).fit(X50[:, sel], y)

    feat = Pipeline([("scaler", scaler), ("pca", pca)])
    feat_blob = to_onnx(
        feat, initial_types=[("input", FloatTensorType([None, 20]))]
    ).SerializeToString()
    svm_blob = to_onnx(
        rbf, initial_types=[("input", FloatTensorType([None, len(sel)]))]
    ).SerializeToString()
    fsess = ort.InferenceSession(feat_blob, providers=["CPUExecutionProvider"])
    ssess = ort.InferenceSession(svm_blob, providers=["CPUExecutionProvider"])
    onnx50 = fsess.run(None, {"input": X.astype(np.float32)})[0]
    onnx_label = ssess.run(None, {"input": onnx50[:, sel].astype(np.float32)})[0]
    sk_label = rbf.predict(X50[:, sel])
    agree = (np.asarray(onnx_label) == sk_label).mean()
    assert agree >= 0.999
