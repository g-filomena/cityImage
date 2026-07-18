"""Sphinx configuration for the cityImage documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import cityImage  # noqa: E402

# -- Project information -----------------------------------------------------

project = "cityImage"
copyright = "2026, Gabriele Filomena"
author = "Gabriele Filomena"
release = getattr(cityImage, "__version__", "1.2.3")
version = release
html_title = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

# -- Intersphinx -------------------------------------------------------------
# Enables validated cross-references to Python objects in this project and to
# external libraries (used from Markdown notebook cells and docstrings).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "osmnx": ("https://osmnx.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
# Do not fail the build if an inventory cannot be fetched (e.g. offline builds).
intersphinx_timeout = 10
intersphinx_disabled_reftypes = ["*"]

templates_path = ["_templates"]
exclude_patterns = [
    "otherExamples/*",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Avoid executing OSM/network-heavy notebooks during documentation builds.
# Notebook runtime smoke tests are handled separately.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/g-filomena/cityImage",
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "navigation_depth": 4,
    "show_prev_next": True,
    "show_toc_level": 2,
    }

html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_typehints = "none"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
