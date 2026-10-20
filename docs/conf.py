"""Sphinx config for the Quobo API reference (Enhancement 9).

Build:  .venv/Scripts/python -m sphinx -b html docs docs/_build/html
Docs are auto-generated from module + function docstrings (autodoc), so they
stay in sync with the code."""

import os
import sys

# Make the `quobo` package importable for autodoc. The package lives under
# src/ (src-layout), so the path entry must be src/ — inserting the repo
# root does NOT make `import quobo` work; it only resolves via an editable
# install (.pth). Inserting src/ means the docs build works on a machine
# that has not run `pip install -e .`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "quobo"
copyright = "2026, IDP3"
author = "IDP3"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

napoleon_numpy_docstring = True
autodoc_typehints = "description"

exclude_patterns = ["_build"]
