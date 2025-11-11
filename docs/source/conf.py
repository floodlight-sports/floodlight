import datetime
import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath("../.."))

import floodlight  # noqa: 402


# -- Project information -----------------------------------------------------
project = "floodlight"
version = floodlight.__version__
year = datetime.datetime.now().year
author = "Dominik Raabe"
copyright = f"{year}, {author}"

release = f"v{version}"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "display_version": True,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "prev_next_buttons_location": None,
    "style_external_links": True,
    "style_nav_header_background": "#006666",
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
