"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from __future__ import annotations

import contextlib
import os
import re

import requests

# -- Project information -----------------------------------------------------
project = "ComPWA benchmarks"
PACKAGE = "benchmarks"
REPO_NAME = "benchmarks"
copyright = "2022, ComPWA"
author = "Common Partial Wave Analysis"


def get_branch_name() -> str:
    branch_name = os.environ.get("READTHEDOCS_VERSION", "stable")
    if branch_name == "latest":
        return "main"
    if re.match(r"^\d+$", branch_name):  # PR preview
        return "stable"
    return branch_name


def get_logo_path() -> str | None:
    path = "_static/logo.svg"
    with contextlib.suppress(requests.exceptions.ConnectionError):
        _fetch_logo(
            url="https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg",
            output_path=path,
        )
    if os.path.exists(path):
        return path
    return None


def get_nb_execution_mode() -> str:
    if "FORCE_EXECUTE_NB" in os.environ:
        print("\033[93;1mWill run ALL Jupyter notebooks!\033[0m")
        return "force"
    if "EXECUTE_NB" in os.environ:
        return "cache"
    return "off"


def _fetch_logo(url: str, output_path: str) -> None:
    if os.path.exists(output_path):
        return
    online_content = requests.get(url, allow_redirects=True)
    with open(output_path, "wb") as stream:
        stream.write(online_content.content)


autosectionlabel_prefix_document = True
codeautolink_concat_default = True
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "  # doctest
default_role = "py:obj"
exclude_patterns = [
    "**.ipynb_checkpoints",
    "*build",
    "adr/template.md",
    "tests",
]
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_togglebutton",
]
html_copy_source = True  # needed for download notebook button
html_favicon = "_static/favicon.ico"
html_last_updated_fmt = "%-d %B %Y"
html_logo = get_logo_path()
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "logo": {"text": project},
    "repository_url": f"https://github.com/ComPWA/{REPO_NAME}",
    "repository_branch": get_branch_name(),
    "path_to_docs": "docs",
    "use_download_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
    },
    "show_navbar_depth": 2,
    "show_toc_level": 2,
}
html_title = project
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
}
linkcheck_anchors = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
]
myst_linkify_fuzzy_links = False
myst_update_mathjax = False
nb_execution_mode = get_nb_execution_mode()
nb_execution_show_tb = True
nb_execution_timeout = -1
nb_output_stderr = "remove"
nitpick_ignore = [
    ("py:class", "ArraySum"),
    ("py:class", "ampform.sympy._array_expressions.MatrixMultiplication"),
    ("py:class", "ipywidgets.widgets.widget_float.FloatSlider"),
    ("py:class", "ipywidgets.widgets.widget_int.IntSlider"),
    ("py:class", "typing_extensions.Protocol"),
]
nitpicky = True  # warn if cross-references are missing
primary_domain = "py"
pygments_style = "sphinx"
