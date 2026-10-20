"""Import-cost guards — the heavy dependencies must stay lazy.

TensorFlow is the project's heaviest dependency: importing it took ~80-190
seconds on the development machine. It used to be imported at module scope in
``src/quobo/features.py``, which meant *every* consumer paid that cost — the
PCA/scaler path, the fingerprint helpers, every CLI that merely imported a
symbol, and every test that imported the pipeline. The whole test suite took
199 seconds, and ~190 of them were this one import.

These tests pin the property so the cost cannot silently come back:

* importing :mod:`src.quobo.features` must not pull TensorFlow into
  ``sys.modules``;
* importing the run pipeline must not pull TensorFlow in either;
* the lazy hook must still expose a real, working ``features.tf``.

They are deliberately cheap — they assert on ``sys.modules`` *after* an
import, so they never pay the TensorFlow cost themselves.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Runs in a fresh interpreter so the check is not polluted by whatever other
# test modules already imported.
_PROBE = """
import sys
sys.path.insert(0, {root!r})
import importlib
for name in {modules!r}:
    importlib.import_module(name)
heavy = [m for m in ("tensorflow", "keras", "torch", "qiskit") if m in sys.modules]
print(",".join(heavy))
"""


def _probe(modules: list[str]) -> set[str]:
    """Import ``modules`` in a fresh interpreter; return the heavy libs loaded."""
    code = _PROBE.format(root=str(ROOT), modules=modules)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, f"probe failed: {out.stderr}"
    return {m for m in out.stdout.strip().split(",") if m}


def test_importing_features_does_not_pull_tensorflow():
    heavy = _probe(["src.quobo.features"])
    assert "tensorflow" not in heavy, (
        "importing src.quobo.features pulled in TensorFlow; the lazy-import "
        "guard in features.py has regressed"
    )


def test_importing_the_run_pipeline_does_not_pull_tensorflow():
    """The pipeline module is imported by most tests; it must stay light."""
    heavy = _probe(["src.quobo.run_experiment"])
    assert "tensorflow" not in heavy, (
        "importing src.quobo.run_experiment pulled in TensorFlow; the "
        "transitive import chain regressed"
    )


def test_importing_the_selection_module_does_not_pull_tensorflow():
    heavy = _probe(["src.quobo.selection", "src.quobo.qsvm"])
    assert "tensorflow" not in heavy


def test_features_tf_attribute_resolves_to_real_tensorflow():
    """The lazy hook must still yield the genuine module.

    ``features.tf`` is a module-level ``__getattr__`` hook (PEP 562). It has
    to return the real TensorFlow, and it has to return the SAME object on
    every call — otherwise a monkeypatch of ``features.tf.keras.utils`` would
    not affect the code that actually calls ``tf.keras.utils.load_img``.
    """
    from src.quobo import features as F

    first = F.tf
    second = F.tf
    assert first is second
    assert first.__name__ == "tensorflow"
    assert hasattr(first.keras.utils, "load_img")


def test_features_rejects_unknown_attributes():
    """The lazy hook must not swallow genuine typos."""
    from src.quobo import features as F

    try:
        F.definitely_not_a_real_attribute
    except AttributeError:
        return
    raise AssertionError("features.__getattr__ should raise AttributeError for unknown names")


def test_load_model_and_extract_embeddings_use_the_shared_module():
    """The in-function TensorFlow handle must be the same object as ``F.tf``.

    If the two resolved to different module objects, patching
    ``F.tf.keras.utils.load_img`` in a test would silently stop affecting the
    extraction path — a failure that would only show up as a slow test, not
    an error.
    """
    from src.quobo import features as F

    assert F._load_tf() is F.tf
