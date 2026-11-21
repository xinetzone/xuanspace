"""测试辅助：临时伪造 FFI 模块的内容与位置描述。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SOURCE = '''\
"""测试用 FFI 模块：覆盖版本、可用性、普通函数、类方法与不可序列化结果。"""

__version__ = "0.0.1"


def is_available() -> bool:
    return True


def version() -> str:
    return "9.9.9"


def add(left: int, right: int) -> int:
    return left + right


class ops:
    @staticmethod
    def mul(left: int, right: int) -> int:
        return left * right


def unsupported() -> set:
    return {1, 2}
'''


@dataclass(frozen=True)
class FakeFFI:
    """临时 FFI 模块的位置与导入名。"""

    directory: Path
    module: str
