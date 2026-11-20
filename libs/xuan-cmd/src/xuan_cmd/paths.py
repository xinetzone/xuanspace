"""工程路径解析：显式参数优先、环境变量兜底，统一转为绝对路径。"""

import os
from os import PathLike
from pathlib import Path

from .errors import XuanCmdError

PathInput = str | PathLike[str]


def _absolute(raw: PathInput, base: PathInput | None) -> Path:
    """展开 ``~``；相对值以 ``base``（缺省为当前目录）为锚点解析为绝对路径。"""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return ((Path(base) if base is not None else Path.cwd()) / path).resolve()


def resolve_dir(
    value: PathInput | None,
    *,
    envvar: str | None = None,
    base: PathInput | None = None,
    label: str = "目录",
    require_exists: bool = True,
) -> Path:
    """解析一个必须由调用方给定的目录。

    取值优先级为 ``value`` > 环境变量 ``envvar``；两者都为空（含空字符串）时抛
    :class:`XuanCmdError`。相对路径以 ``base``（缺省为当前目录）为锚点解析。
    ``require_exists`` 为真时断言结果确实是目录。
    """
    raw = value or (os.environ.get(envvar) if envvar else None)
    if not raw:
        message = f"未提供{label}"
        if envvar:
            message += f"：请显式传入或设置环境变量 {envvar}"
        raise XuanCmdError(message)
    path = _absolute(raw, base)
    if require_exists and not path.is_dir():
        raise XuanCmdError(f"{label}不存在：{path}")
    return path


def resolve_work_dir(value: PathInput | None, default: PathInput, *, base: PathInput | None = None) -> Path:
    """解析工作目录：显式值按 ``base`` 解析为绝对路径，缺省时原样返回 ``default``。

    ``default`` 由调用方给出（通常是某个工程目录下的固定子目录），本函数不对其
    做存在性校验，也不改写其相对性。
    """
    if value:
        return _absolute(value, base)
    return Path(default)
