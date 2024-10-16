from sphinx_api_relink.helpers import get_branch_name, get_execution_mode

BRANCH = get_branch_name()
ORGANIZATION = "ComPWA"
PACKAGE = "benchmarks"
REPO_NAME = "benchmarks"
REPO_TITLE = "ComPWA benchmarks"


autosectionlabel_prefix_document = True
codeautolink_concat_default = True
copybutton_prompt_is_regexp = True
copybutton_prompt_text = r">>> |\.\.\. "
copyright = f"2024, {ORGANIZATION}"  # noqa: A001
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
html_logo = (
    "https://raw.githubusercontent.com/ComPWA/ComPWA/04e5199/doc/images/logo.svg"
)
html_show_copyright = False
html_show_sourcelink = False
html_show_sphinx = False
html_sourcelink_suffix = ""
html_static_path = ["_static"]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "Common Partial Wave Analysis",
            "url": "https://compwa.github.io",
            "icon": "https://compwa.github.io/_static/favicon.ico",
            "type": "url",
        },
        {
            "name": "GitHub",
            "url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Launch on Binder",
            "url": f"https://mybinder.org/v2/gh/{ORGANIZATION}/{REPO_NAME}/{BRANCH}?filepath=docs",
            "icon": "https://mybinder.readthedocs.io/en/latest/_static/favicon.png",
            "type": "url",
        },
        {
            "name": "Launch on Colaboratory",
            "url": f"https://colab.research.google.com/github/{ORGANIZATION}/{REPO_NAME}/blob/{BRANCH}",
            "icon": "https://avatars.githubusercontent.com/u/33467679?s=100",
            "type": "url",
        },
    ],
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "deepnote_url": "https://deepnote.com",
        "notebook_interface": "jupyterlab",
    },
    "logo": {"text": REPO_TITLE},
    "path_to_docs": "docs",
    "repository_branch": BRANCH,
    "repository_url": f"https://github.com/{ORGANIZATION}/{REPO_NAME}",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
}
html_title = REPO_TITLE
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
nb_execution_mode = get_execution_mode()
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
project = REPO_TITLE
