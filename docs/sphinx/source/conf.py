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
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_rtd_theme",
]

master_doc = "index"
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
templates_path = ["_templates"]
exclude_patterns = [
    'build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
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
autosummary_generate_overwrite = True
autosummary_imported_members = True

# -- nbsphinx configuration (Jupyter notebooks) ------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 300

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

# -- Bibliography configuration (sphinxcontrib-bibtex) -----------------------
import pybtex.plugin
from pybtex.richtext import Text
from pybtex.style.formatting.alpha import Style as _AlphaStyle
from pybtex.style.names import BaseNameStyle


class _LastInitialNameStyle(BaseNameStyle):
    """Format names as 'Last AB' (no dots, initials together)."""

    def format(self, person, abbr=False):
        parts = []
        for name in person.rich_prelast_names:
            parts.extend([name, ' '])
        for i, name in enumerate(person.rich_last_names):
            if i > 0:
                parts.append(' ')
            parts.append(name)
        initials = ''.join(
            n[0] for n in
            person.first_names + person.middle_names
            if n
            )
        if initials:
            parts.extend([' ', initials])
        if person.rich_lineage_names:
            parts.append(', ')
            for name in person.rich_lineage_names:
                parts.append(name)
        return Text(*parts)


class _ProteopyStyle(_AlphaStyle):
    default_name_style = 'last_initial'


pybtex.plugin.register_plugin(
    'pybtex.style.names', 'last_initial',
    _LastInitialNameStyle,
    )
pybtex.plugin.register_plugin(
    'pybtex.style.formatting', 'proteopy',
    _ProteopyStyle,
    )

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "proteopy"
bibtex_reference_style = "author_year"
