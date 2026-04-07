# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(".."))

import mapca

# -- Project information -----------------------------------------------------
project = "mapca"
copyright = "2020, mapca developers"
author = "mapca developers"
release = mapca.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "nilearn": ("https://nilearn.github.io/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
