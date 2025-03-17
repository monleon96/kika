import datetime
import os
import sys

# Add the repository root directory to sys.path
sys.path.insert(0, os.path.abspath('../..'))

# Attempt safe import
try:
    from mcnpy._config import LIBRARY_VERSION, AUTHOR
except ImportError:
    LIBRARY_VERSION = "unknown"
    AUTHOR = "unknown"
    print("WARNING: Could not import LIBRARY_VERSION or AUTHOR from mcnpy._config.")

project = 'MCNPy'
copyright = f"{datetime.datetime.now().year}, {AUTHOR}"
author = AUTHOR
release = LIBRARY_VERSION

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Autodoc configuration
autodoc_default_options = {
    'no-index': True,
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'undoc-members': True,
}

# -- HTML output options -----------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3,
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_with_keys': True,
    'logo_only': False,
    'style_external_links': True,
    'includehidden': True,
}

html_static_path = ['_static']

html_title = "MCNPy Documentation"
html_short_title = "MCNPy"

html_sidebars = {
    '**': [
        'globaltoc.html',
        'searchbox.html',
        'relations.html',
    ]
}

rst_prolog = f"""
.. |version| replace:: {LIBRARY_VERSION}
"""
