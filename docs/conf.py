"""Sphinx config for the Quobo API reference (Enhancement 9).

Build:  .venv/Scripts/python -m sphinx -b html docs docs/_build/html
Docs are auto-generated from module + function docstrings (autodoc), so they
stay in sync with the code."""

import os
import sys

# make the src/ package importable for autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
