# -- Project information -----------------------------------------------------
import datetime
from importlib.metadata import version as _version

project = 'CausalDisco'
author = 'Alexander G. Reisach, Sebastian Weichwald'
copyright = f'2021-{datetime.date.today().year}, {author}'
# single source of truth for the version is pyproject.toml
release = _version('CausalDisco')
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_logo = 'logo.svg'
html_favicon = 'logo.svg'
html_theme = 'sphinx_rtd_theme'
html_show_sphinx = False
