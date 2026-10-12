# SPDX-License-Identifier: GPL-2.0-only
"""
xuan_compose - Compose 规范声明式容器编排引擎。

由 containers/podman-compose v1.6.0（GPL-2.0-only）分层重构而来，
行为与上游 1.6.0 对等，包结构按执行层/规范层/翻译层/引擎层/命令层/CLI 层拆分。

库的使用方式是显式构造引擎，导入本包不产生任何副作用（不读取解释器启动参数、
不创建全局引擎实例、不发起子进程或网络访问；argv 读取仅在 CLI 层）。

版本守卫（P2-10）：本库依赖 PEP 649 的注解惰性求值（3.14 起），低版本
解释器上类体内自引用注解会在定义时抛出晦涩的 ``NameError``。守卫位于本
文件最顶部、先于一切第三方依赖导入，使 PYTHONPATH 直跑/IDE 误用低版本
Python 时得到可诊断的 ``RuntimeError``。本函数刻意只用各版本通用的语法
书写，保证守卫自身在旧解释器上可执行。
"""

import sys

MINIMUM_PYTHON_VERSION = (3, 14)


def require_python_version(minimum: tuple = MINIMUM_PYTHON_VERSION, current=None) -> None:
    """校验运行解释器版本；低于 ``minimum`` 时抛友好 ``RuntimeError``。

    ``current`` 仅用于测试注入（默认取 ``sys.version_info``）。元组按字典序
    比较，补丁版本不影响判定：``(3, 14, 0)`` 满足下限 ``(3, 14)``。
    """
    running = sys.version_info if current is None else current
    if running < minimum:
        raise RuntimeError(
            f"xuan_compose requires Python {minimum[0]}.{minimum[1]} or newer, "
            f"but the running interpreter is Python {running[0]}.{running[1]}. "
            "This release depends on PEP 649 deferred annotation evaluation; "
            f"pyproject.toml declares requires-python >= {minimum[0]}.{minimum[1]}. "
            "Please run xuan-compose with a supported Python interpreter."
        )


require_python_version()

__version__ = "1.6.0+xuan.1"
