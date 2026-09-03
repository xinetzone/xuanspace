#!/usr/bin/env python3
"""Sphinx configuration file for the 'Xuanspace' project documentation."""

import importlib.util as _ilut
import os
from pathlib import Path

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
    "mystx",
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


if _has("mystx"):
    html_theme = "mystx"
elif _has("sphinx_book_theme"):
    html_theme = "sphinx_book_theme"
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
    "python": ("https://docs.python.org/3.14", None),
}


copybutton_exclude = ".linenos, .gp"
copybutton_selector = ":not(.prompt) > div.highlight pre"


html_theme_options = {
    "repository_url": "https://github.com/xinetzone/xuanspace",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "doc",
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

# linkify 需要 linkify-it-py 可选依赖，缺失时降级跳过
try:
    import linkify_it  # noqa: F401
except ImportError:
    myst_enable_extensions = [e for e in myst_enable_extensions if e != "linkify"]

myst_heading_anchors = 3
myst_footnote_transition = False
myst_heading_slug_func = "docutils.nodes.make_id"

myst_fence_as_directive = ["mermaid"]

suppress_warnings = ["myst.xref_missing"]

templates_path = ["_templates"]


napoleon_use_ivar = True


mermaid_params = ["--theme", "default"]
mermaid_version = "10.9.0"


def _has_sphinx_app():
    """Delayed import so conf.py can still be read by pure config parsers."""
    try:
        from sphinx.application import Sphinx  # noqa: F401
        return True
    except Exception:
        return False


def setup(app):
    """Xuanspace project setup hook — declare parallel safety & add guard.

    P1-A 审计结论（2026-09-03）：当前启用扩展的 parallel_{read,write}_safe 元数据
    均为 True，无覆写需求。将来启用 autoapi / _ext.gallery_directive 等上游未声
    明扩展时，将其加入下方覆写表即可，无需去上游提 PR。
    """
    _PARALLEL_SAFE_OVERRIDES: dict[str, tuple[bool, bool]] = {
        # "autoapi.extension":        (True, True),
        # "sphinx.ext.linkcode":      (True, True),
    }
    for ext_name, (pr, pw) in _PARALLEL_SAFE_OVERRIDES.items():
        if ext_name in getattr(app, "extensions", {}):
            setattr(app.extensions[ext_name], "parallel_read_safe", bool(pr))
            setattr(app.extensions[ext_name], "parallel_write_safe", bool(pw))

    def _assert_parallel_ready(app, env):
        if app.parallel > 0 and not app.is_parallel_allowed("read"):
            raise RuntimeError(
                "Parallel read disabled. Check extensions missing "
                "parallel_read_safe=True. Re-run with -v to see per-extension "
                "warnings from Sphinx.is_parallel_allowed()."
            )

    if _has_sphinx_app():
        app.connect("env-before-read-docs", _assert_parallel_ready)

    return {
        "version": version,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
