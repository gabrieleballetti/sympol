# Configuration file for the Sphinx documentation builder.
#
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "sympol"
copyright = "2023, Gabriele Balletti"
author = "Gabriele Balletti"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
# numpydoc_show_class_members = False
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
html_context = {
    "sidebar_external_links_caption": "Links",
    "sidebar_external_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source code',
            "https://github.com/gabrieleballetti/sympol",
        ),
        (
            '<i class="fa fa-bug fa-fw"></i> Issue tracker',
            "https://github.com/gabrieleballetti/sympol/issues",
        ),
    ],
}
