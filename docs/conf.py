# -- Project information -----------------------------------------------------
import datetime
# import CausalDisco  # needed for `.. automodule:: in the .rst`

project = 'CausalDisco'
author = 'Alexander G. Reisach, Sebastian Weichwald'
copyright = f'2021-{datetime.date.today().year}, {author}'
version = '0.2.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
]

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_logo = 'logo.png'
html_favicon = 'logo.png'
html_theme = 'sphinx_rtd_theme'
html_show_sphinx = False
