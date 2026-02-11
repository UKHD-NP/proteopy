project = "ProteoPy"
copyright = (
    "2025, BludauLab Neuropathology Heidelberg, "
    "Ian Dirk Fichtner, Isabell Bludau"
    )
author = (
    "Ian Dirk Fichtner, Isabell Bludau, "
    "BludauLab Neuropathology Heidelberg"
    )
version = "0.1.0"
release = "0.1.0"


# -- Path setup --------------------------------------------------------------
import sys
from pathlib import Path

# Add project root directory for autodoc to find proteopy package
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


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

# Bug fix: Sphinx 9.x introduced autosummary.import_cycle detection bug
suppress_warnings = ['autosummary.import_cycle']  

# -- Custom roles ------------------------------------------------------------
rst_prolog = """
.. role:: blue-bold
"""

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_logo = "../../logos/logo_colour.png"
html_favicon = "../../logos/favicon_colour.png"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# Custom CSS and JS files
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Napoleon configuration (NumPy docstrings) -------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True

# -- Intersphinx configuration (cross-references) ----------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    # "mudata": ("https://mudata.readthedocs.io/en/latest/", None),  # Currently unavailable
}
