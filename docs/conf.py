#!/usr/bin/env python3
"""Sphinx configuration file for the 'Xuanspace' project documentation."""

import importlib.util as _ilut
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    import asyncio

    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


ROOT = Path(__file__).resolve().parents[1]


project = "Xuanspace（玄境）"
author = "xinetzone"
copyright = "2026, xinetzone"
release = "0.1.0"
version = release

language = "zh_CN"


def _has(mod: str) -> bool:
    try:
        return _ilut.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


core_exts = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
]

optional_exts = [
    "myst_parser",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_book_theme",
]

extensions = core_exts.copy()
extensions.extend([e for e in optional_exts if _has(e)])


exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    "README.md",
]

master_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

html_last_updated_fmt = "%Y-%m-%d, %H:%M:%S"


if _has("sphinx_book_theme"):
    html_theme = "sphinx_book_theme"
elif _has("sphinx_rtd_theme"):
    html_theme = "sphinx_rtd_theme"
else:
    html_theme = "alabaster"

html_title = "Xuanspace（玄境）"
html_copy_source = False

html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

html_compact_lists = True

pygments_style = "sphinx"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3.13", None),
}


copybutton_exclude = ".linenos, .gp"
copybutton_selector = ":not(.prompt) > div.highlight pre"


html_theme_options = {
    "repository_url": "https://github.com/xinetzone/xuanspace",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "max_navbar_depth": 3,
    "collapse_navbar": False,
}


html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://localhost:8000/")


numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3
myst_footnote_transition = False
myst_heading_slug_func = "docutils.nodes.make_id"

myst_fence_as_directive = ["mermaid"]

suppress_warnings = ["myst.xref_missing"]

templates_path = ["_templates"]


napoleon_use_ivar = True


mermaid_params = ["--theme", "default"]
mermaid_version = "10.9.0"
