#!/usr/bin/env python3
"""Sphinx configuration for the 'xuan-compose' documentation.

遵循 xuanspace monorepo 文档体系约定（参照根 doc/conf.py），按 lib 级惯例
裁剪（参照 libs/mystx、libs/tvm-book）：纯 Markdown 文档集，无
Makefile/make.bat（本地手动 ``sphinx-build``；主仓 ``xs doc`` 仅构建根
doc/，不覆盖本目录）。
"""

import importlib.util as _ilut
import os

project = "xuan-compose"
author = "Xuanspace Team"
copyright = "2026, Xuanspace Team"
release = "0.1.0"
version = release

language = "zh_CN"


def _has(mod: str) -> bool:
    try:
        return _ilut.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


# myst_parser 为必需扩展：文档集是纯 Markdown，缺失时明确失败优于静默跳过。
extensions = ["myst_parser"]

# 其余扩展按环境降级：本地/CI/读者环境差异下构建不中断。
optional_exts = [
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",  # dev/delivery 证据页含 mermaid 图
]
extensions.extend([e for e in optional_exts if _has(e)])

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"
source_suffix = {".md": "markdown"}

if _has("mystx"):
    html_theme = "mystx"
elif _has("sphinx_book_theme"):
    html_theme = "sphinx_book_theme"
else:
    html_theme = "alabaster"

# 主题选项仅对 mystx/sphinx_book_theme 有效；alabaster 下声明未知键会告警。
if _has("mystx") or _has("sphinx_book_theme"):
    html_theme_options = {
        "repository_url": "https://github.com/xinetzone/xuanspace",
        "use_repository_button": True,
        "use_issues_button": True,
        "use_edit_page_button": True,
        "repository_branch": "main",
        "path_to_docs": "libs/xuan-compose/doc",
        "home_page_in_toc": True,
        "show_navbar_depth": 2,
        "max_navbar_depth": 3,
        "collapse_navbar": False,
    }

html_title = "xuan-compose 文档"
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
html_compact_lists = True
pygments_style = "sphinx"
html_last_updated_fmt = "%Y-%m-%d, %H:%M:%S"
html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://localhost:8000/")

numfig = True

# 不启用 dollarmath/amsmath：文档含 ``$ git ...`` shell 提示符，
# 美元符号会被误解析为数学公式。

myst_heading_anchors = 3
myst_heading_slug_func = "docutils.nodes.make_id"

# 仅在安装 sphinxcontrib.mermaid 时把 ```mermaid 围栏转为指令；
# 缺失时按普通代码块渲染，避免未知指令错误。
if _has("sphinxcontrib.mermaid"):
    myst_fence_as_directive = ["mermaid"]
    mermaid_params = ["--theme", "default"]
    mermaid_version = "10.9.0"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.14", None),
}

copybutton_exclude = ".linenos, .gp"
copybutton_selector = ":not(.prompt) > div.highlight pre"

# myst.xref_missing：跨页引用由链接门禁（check_doc_links.py）校验，构建期放宽。
# toc.not_included 不抑制：全部页面经三层 index 的 :hidden: toctree 收录
# （index → dev/index → 各报告页；dev/parity/index → 三分册），手写导航表负责
# 「按意图阅读」，toctree 负责机器站点结构（侧边栏/上下页/漏收录检测），二者并存。
# 新增页面若未挂进任何 toctree，构建告警即为有效信号，不得抑制。
# misc.highlighting_failure：sphinxcontrib.mermaid 缺失时，```mermaid 围栏按
# 普通代码块渲染（mermaid_params 不再生效），Pygments 无 mermaid 词法器；
# 此为既定的优雅降级，抑制以避免环境污染下的噪声告警。
suppress_warnings = [
    "myst.xref_missing",
    "misc.highlighting_failure",
]
