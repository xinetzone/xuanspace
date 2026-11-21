"""FFI 能力键语法：``xuan:ffi/<name>#api``。

键是跨发行包的能力身份：调用方只依赖键，不依赖具体 FFI 库的模块名，
因此把 ``caffe_ffi`` 换成另一个实现（或换传输模式）不需要改调用方。
"""

import re

_PREFIX = "xuan:ffi/"
_SUFFIX = "#api"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


class FFIKeyError(ValueError):
    """能力键或 FFI 名称不合约。"""


def ffi_key(name: str) -> str:
    """把 FFI 名称拼成能力键，并校验名称语法。

    名称须与小写字母开头，只含小写字母、数字、下划线与连字符——与
    ``xuan_plugin.manifest`` 的能力键正则保持一致。
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise FFIKeyError(f"FFI 名称 {name!r} 不合约，期望小写字母开头，允许数字、下划线与连字符")
    return f"{_PREFIX}{name}{_SUFFIX}"


def ffi_name(key: str) -> str | None:
    """从能力键取回 FFI 名称；不是 ``xuan:ffi/<name>#api`` 形态时返回 None。"""
    if not isinstance(key, str) or not key.startswith(_PREFIX) or not key.endswith(_SUFFIX):
        return None
    name = key[len(_PREFIX) : -len(_SUFFIX)]
    return name if _NAME_RE.match(name) else None
