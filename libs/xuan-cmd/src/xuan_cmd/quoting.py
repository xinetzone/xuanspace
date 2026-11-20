"""跨平台参数引用：让含空格或特殊字符的路径在拼接后仍是单个参数。"""

import os
import shlex
import subprocess
from collections.abc import Iterable
from os import PathLike


def quote_arg(part: str | PathLike[str]) -> str:
    """按当前平台给单个参数加引号。

    Windows 走 ``subprocess.list2cmdline``（MSVC 参数规则），其余平台走
    ``shlex.quote``（POSIX shell 规则）。
    """
    text = os.fspath(part)
    if os.name == "nt":
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def format_command(parts: Iterable[str | PathLike[str]]) -> str:
    """把参数序列拼成一条命令字符串，可直接交给 shell 执行。"""
    return " ".join(quote_arg(part) for part in parts)
