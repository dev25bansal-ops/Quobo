"""Enhancement: ONNX export for edge inference.

Exports the edge-classifier as composable ONNX models that exactly match the
leak-free experiment path (raw 1280-d backbone -> StandardScaler -> PCA(50)
-> select-8 -> RBF-SVM). Two artifacts:

    features_1280_50.onnx   StandardScaler + PCA(50): 1280-d -> 50-d
    head_svm_8.onnx         RBF-SVM trained on the 8 selected components: 8-d -> label

The 8-column selection (fixed indices from QUBO) is applied in numpy by the
inference helper (trivial), because skl2onnx has no reliable column-index op.
This mirrors the experiment exactly: the SVM is trained on `pca_feats[:, sel]`.

The MobileNetV2 backbone (224x224x3 -> 1280) is the heavy piece and is exported
separately with --backbone (needs tf2onnx).

Full edge chain: image -> backbone -> features_1280_50 -> [x50[:, sel]] -> head_svm_8.

Usage:
    .venv/Scripts/python.exe -m scripts.export_onnx --config configs/experiment_6class.yaml
    .venv/Scripts/python.exe -m scripts.export_onnx --config ... --backbone
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quobo.config import ROOT, load_config
from src.quobo.features import fit_pca_on_train, run_raw_embeddings
from src.quobo.run_experiment import crop_image_group
from src.quobo.selection import select_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("export_onnx")

OUT = ROOT / "results" / "onnx"


def _save_onnx(model, initial_type: str, dims: tuple, name: str) -> bytes:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    # no zipmap option: default (zipmap=True) returns [label, zipmap] and
    # works for both the feature Pipeline and the SVM (label is output[0])
    onx = to_onnx(model, initial_types=[(initial_type, FloatTensorType(list(dims)))])
    blob = onx.SerializeToString()
    with open(OUT / name, "wb") as f:
        f.write(blob)
    log.info("exported %s (%d bytes)", name, len(blob))
    return blob


def build_head(cfg: dict) -> dict:
    """Fit + export + verify the edge head. Returns the manifest."""
    import onnxruntime as ort
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    OUT.mkdir(parents=True, exist_ok=True)

    raw = run_raw_embeddings(cfg)
    emb_cols = [c for c in raw.columns if c.startswith("e")]
    X_raw = raw[emb_cols].values.astype(np.float32)
    y = np.asarray(raw["label"].values)
    groups = np.array([crop_image_group(p) for p in raw["crop_path"]])
    n_comp = cfg["features"]["pca_components"]
    k = cfg["selection"]["k"]
    seed = cfg["seed"]

    # deterministic reference split (rep 0) — the deployed head
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["qsvm"]["test_fraction"], random_state=seed)
    idx_tr, idx_te = next(gss.split(X_raw, y, groups=groups))
    Xtr, Xte = X_raw[idx_tr], X_raw[idx_te]
    ytr = y[idx_tr]
    Xtr_p, _Xall_p, ev_pct = fit_pca_on_train(Xtr, np.vstack([Xtr, Xte]), n_comp, seed)

    sel, sel_info = select_features("D_qubo", Xtr_p, ytr, k, seed, cfg["qubo"])

    # --- feature pipeline (must match fit_pca_on_train: scaler -> pca) ---
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=n_comp, random_state=seed).fit(scaler.transform(Xtr))
    Xtr50 = pca.transform(scaler.transform(Xtr))
    Xte50 = pca.transform(scaler.transform(Xte))
    assert np.allclose(Xtr50, Xtr_p, atol=1e-4), "export PCA != experiment PCA"

    # --- SVM head trained on the 8 selected components ---
    rbf = SVC(kernel="rbf", class_weight="balanced", random_state=seed).fit(Xtr50[:, sel], ytr)
    classes = sorted({str(v) for v in y})

    # feature pipeline: 1280-d -> 50-d (StandardScaler + PCA as a Pipeline)
    from sklearn.pipeline import Pipeline
    feat_pipe = Pipeline([("scaler", scaler), ("pca", pca)])
    _save_onnx(feat_pipe, "input", (None, X_raw.shape[1]), "features_1280_50.onnx")
    # svm head: 8-d -> label
    _save_onnx(rbf, "input", (None, k), "head_svm_8.onnx")

    # --- parity: full head chain vs sklearn on the test set ---
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType
    feat_blob = to_onnx(feat_pipe,
                        initial_types=[("input", FloatTensorType([None, X_raw.shape[1]]))],
                        ).SerializeToString()
    svm_blob = to_onnx(rbf, initial_types=[("input", FloatTensorType([None, k]))]).SerializeToString()
    feat_sess = ort.InferenceSession(feat_blob, providers=["CPUExecutionProvider"])
    svm_sess = ort.InferenceSession(svm_blob, providers=["CPUExecutionProvider"])
    onnx_50 = feat_sess.run(None, {"input": Xte.astype(np.float32)})[0]
    onnx_label = svm_sess.run(None, {"input": onnx_50[:, sel].astype(np.float32)})[0]
    sk_label = rbf.predict(Xte50[:, sel])
    agree = float((onnx_label == sk_label).mean())
    log.info("head parity: %.2f%% (%d/%d)", agree * 100,
             (onnx_label == sk_label).sum(), len(sk_label))

    return {
        "k": k,
        "n_pca_components": n_comp,
        "selected_features": sel,
        "classes": classes,
        "seed": seed,
        "alpha": sel_info.get("alpha"),
        "explained_variance_pct": round(ev_pct, 2),
        "head_parity_pct": round(agree * 100, 2),
        "backbone_dim": X_raw.shape[1],
        "artifacts": ["features_1280_50.onnx", "head_svm_8.onnx",
                      "inference.py", "export_manifest.json"],
    }


def try_backbone(cfg: dict) -> bool:
    """Export MobileNetV2 (224x224x3 -> 1280) via tf2onnx; best-effort."""
    try:
        import tensorflow as tf
        import tf2onnx

        from src.quobo.features import load_model
        model = load_model()
        spec = [tf.TensorSpec([1, 224, 224, 3], tf.float32, name="image")]
        conv = tf2onnx.convert.from_keras(model, input_signature=spec,
                                          output_path=str(OUT / "backbone_mobilenet.onnx"))
        # from_keras returns (onnx_model, inputs, outputs) — the model is [0]
        onnx_model = conv[0] if isinstance(conv, tuple) else conv
        log.info("exported backbone_mobilenet.onnx (%d bytes)", len(onnx_model.SerializeToString()))
        return True
    except Exception as e:  # noqa: BLE001 — best-effort backbone export; any failure skips it
        log.warning("backbone export skipped (tf2onnx unavailable/failed): %s: %s",
                    type(e).__name__, str(e)[:120])
        return False


def write_inference_helper() -> None:
    (OUT / "inference.py").write_text('''"""Standalone edge inference for the Quobo ONNX classifier head.

Chain (matches the leak-free experiment exactly):
    image -> backbone_mobilenet.onnx (224x224x3 -> 1280)  [optional/heavy]
          -> features_1280_50.onnx (1280 -> 50)           [StandardScaler+PCA]
          -> x50[:, selected_features]                    [numpy index, 8 cols]
          -> head_svm_8.onnx (8 -> label)
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent


def _session(name: str):
    return ort.InferenceSession(str(HERE / name), providers=["CPUExecutionProvider"])


def load():
    manifest = json.loads((HERE / "export_manifest.json").read_text())
    return manifest


def predict(x1280: np.ndarray) -> str:
    """x1280: (1, 1280) backbone output (float32). Returns predicted class."""
    m = load()
    x50 = _session("features_1280_50.onnx").run(None, {"input": x1280.astype(np.float32)})[0]
    label = _session("head_svm_8.onnx").run(
        None, {"input": x50[:, m["selected_features"]].astype(np.float32)})[0]
    # zipmap=True: output[0] is already the class-name string
    return str(label[0])
''', encoding="utf-8")
    log.info("wrote inference.py helper")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_6class.yaml")
    ap.add_argument("--backbone", action="store_true",
                    help="also export the MobileNetV2 backbone (needs tf2onnx)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_head(cfg)
    manifest["backbone_exported"] = try_backbone(cfg) if args.backbone else False
    if manifest["backbone_exported"]:
        manifest["artifacts"] = manifest["artifacts"] + ["backbone_mobilenet.onnx"]
    write_inference_helper()
    with open(OUT / "export_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("done. head parity %.2f%%, backbone=%s",
             manifest["head_parity_pct"], manifest["backbone_exported"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
