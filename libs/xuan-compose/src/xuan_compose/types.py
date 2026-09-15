# SPDX-License-Identifier: GPL-2.0-only
"""跨层共享的类型别名与枚举。

上游 1.6.0 全程使用裸 ``dict`` 表示服务，这里仅提供更精确的注解别名，
运行时结构仍是 dict，禁止改为 dataclass（compose 文档允许任意扩展键）。
"""

from enum import StrEnum
from typing import Any

#: compose 服务定义：文档中的 ``services.<name>`` 映射，允许任意扩展键。
Service = dict[str, Any]


class DependField(StrEnum):
    """服务依赖图在服务 dict 中的内部键（迁移自上游 1.6.0 第 3790-3792 行）。

    上游写作 ``class DependField(str, Enum)``；Python 3.14 下按 ruff UP042
    现代化为 ``StrEnum``（差异表登记）。全部使用点均为 dict 键查找，
    等值/哈希语义不变。
    """

    DEPENDENCIES = "_deps"
    DEPENDENTS = "_dependents"
