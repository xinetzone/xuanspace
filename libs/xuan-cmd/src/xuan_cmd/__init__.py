"""Xuan-Cmd：命令装配与工程路径解析工具库。"""

from .errors import XuanCmdError
from .paths import resolve_dir, resolve_work_dir
from .projects import DEFAULT_MARKER, list_projects, locate_project
from .quoting import format_command, quote_arg

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_MARKER",
    "XuanCmdError",
    "__version__",
    "format_command",
    "list_projects",
    "locate_project",
    "quote_arg",
    "resolve_dir",
    "resolve_work_dir",
]
