# SPDX-License-Identifier: GPL-2.0-only
"""跨平台与基础标量辅助函数。

迁移自 podman_compose.py 的顶层辅助函数（上游 1.6.0 第 47-159、565-579 行）。
双平台路径判定（posix 上用 ntpath.isabs、nt 上用 posixpath.isabs）为上游刻意
设计，必须原样保留：compose 文件可能在另一种 OS 上生成，需要同时按两套规则判绝对路径。
"""

import os
import re
from collections.abc import Iterable
from typing import overload

# Python 按当前 OS 加载对应的 path 模块，但我们需要同时按两大 OS 的规则判定绝对路径。
if os.name == "posix":
    from ntpath import isabs as secondarypathisabs
if os.name == "nt":
    from posixpath import isabs as secondarypathisabs

__all__ = [
    "secondarypathisabs",
    "is_list",
    "is_relative_ref",
    "filteri",
    "try_int",
    "try_float",
    "str_to_seconds",
    "ver_as_list",
    "strverscmp_lt",
    "try_parse_bool",
]

num_split_re = re.compile(r"(\d+|\D+)")

t_re = re.compile(r"^(?:(\d+)[m:])?(?:(\d+(?:\.\d+)?)s?)?$")
STOP_GRACE_PERIOD = "10"


def is_list(list_object: object) -> bool:
    return (
        not isinstance(list_object, str)
        and not isinstance(list_object, dict)
        and hasattr(list_object, "__iter__")
    )


def is_relative_ref(path: str) -> bool:
    return (
        path.startswith("./")
        or path.startswith(".:")
        or path.startswith("../")
        or path.startswith("..:")
    )


# identity filter
def filteri(a: Iterable[str | None]) -> list[str]:
    return [i for i in a if i]


@overload
def try_int(i: int | str, fallback: int) -> int: ...
@overload
def try_int(i: int | str, fallback: None) -> int | None: ...


def try_int(i: int | str, fallback: int | None = None) -> int | None:
    try:
        return int(i)
    except (ValueError, TypeError):
        return fallback


def try_float(i: int | str, fallback: float | None = None) -> float | None:
    try:
        return float(i)
    except (ValueError, TypeError):
        return fallback


def str_to_seconds(txt: int | str | None) -> int | float | None:
    if not txt:
        return None
    if isinstance(txt, (int, float)):
        return txt
    match = t_re.match(txt.strip())
    if not match:
        return None
    mins, sec = match[1], match[2]
    mins = int(mins) if mins else 0
    sec = float(sec) if sec else 0
    # "podman stop" takes only int
    # Error: invalid argument "3.0" for "-t, --time" flag: strconv.ParseUint: parsing "3.0":
    # invalid syntax
    return int(mins * 60.0 + sec)


def ver_as_list(a: str) -> list[int | str]:
    return [try_int(i, i) for i in num_split_re.findall(a)]


def strverscmp_lt(a: str, b: str) -> bool:
    a_ls = ver_as_list(a or "")
    b_ls = ver_as_list(b or "")
    return a_ls < b_ls


def try_parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower()
        if value in ("true", "1"):
            return True
        if value in ("false", "0"):
            return False
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    return None
