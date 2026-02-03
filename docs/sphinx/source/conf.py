project = "ProteoPy"
copyright = (
    "2025, BludauLab Neuropathology Heidelberg, "
    "Ian Dirk Fichtner, Isabell Budau"
    )
author = (
    "Ian Dirk Fichtner, Isabell Budau, "
    "BludauLab Neuropathology Heidelberg"
    )
version = "0.1.0"
release = "0.1.0"


# -- Path setup --------------------------------------------------------------
import os
import sys

# Add source code directory for autodoc
sys.path.insert(0, os.path.abspath("../../../proteopy"))


# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_rtd_theme",
]

master_doc = "index"
source_suffix = '.rst'
templates_path = ["_templates"]
exclude_patterns = [
    'build',
    'Thumbs.db',
    '.DS_Store',
]
language = 'en'

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_logo = "../logos/logo_colour.png"
html_favicon = "l../logos/favicon_color.png"
html_theme_options = {"logo_only": True}

autodoc_member_order = "bysource"

napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autosummary_generate = True
intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy":  ("https://scanpy.readthedocs.io/en/stable/", None),
}
