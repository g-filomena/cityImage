# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cityImage'
copyright = '2024, Gabriele Filomena'
author = 'Gabriele Filomena'
release = '0.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
extensions = [
    'nbsphinx',  # for Jupyter Notebook support
    'sphinx.ext.mathjax',  # for math rendering
    'sphinx.ext.viewcode',  # to include links to source code
    'recommonmark',  # For Markdown files
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = 'pydata_sphinx_theme'

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "github_url": "https://github.com/cityImage",
    "twitter_url": "https://twitter.com/gfilo",
}


html_theme_options = {
    'navigation_depth': 4,
    'show_prev_next': False,
    'show_toc_level': 2,
    'page_sidebar_items': ['page-toc', 'edit-this-page', 'page-nav'],  # Adjust as needed
}

html_static_path = ['_static']


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']