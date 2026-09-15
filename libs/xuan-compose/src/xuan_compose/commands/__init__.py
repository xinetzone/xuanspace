# SPDX-License-Identifier: GPL-2.0-only
"""命令层（T7）——24 个命令 handler 的显式注册表。

上游 podman_compose.py（第 3280-3316 行）用 ``@cmd_run`` 装饰器在类构造期
把 handler 挂到 ``self.commands``；分层后改为模块级显式映射
``COMMAND_HANDLERS``：

1. 导入本包不触碰任何引擎实例、不发起子进程（TR-7.2）；
2. ``COMMAND_HANDLERS`` 与上游 24 个 ``@cmd_run`` 注册名集合相等（TR-7.1）；
3. handler 签名统一为 ``async def handler(compose, args) -> Any``，
   ``compose`` 为 ``Any`` 鸭子类型（与翻译/引擎层惯例一致，T10 差异表）；
4. help/desc 文本与 argparse 装配属 CLI parser 层（T8），不在本注册表。

文件切分对应上游行段：
- ``version``：3378-3388；``systemd``：3411-3513；
- ``pullpush``：3516-3546；``build``：3688-3764；
- ``updown``：802-835、3767-3942、4109-4526；
- ``lifecycle``：3400-3408、4708-4755、4817-4868；
- ``runexec``：4549-4681；``inspect``：3324-3375、4530-4541、4795-4935；
- ``logs``：4759-4792。
"""

import argparse
from collections.abc import Awaitable, Callable
from typing import Any

from .build import compose_build
from .inspect import (
    compose_config,
    compose_images,
    compose_port,
    compose_ps,
    compose_stats,
    list_running_projects,
)
from .lifecycle import (
    compose_kill,
    compose_pause,
    compose_restart,
    compose_start,
    compose_stop,
    compose_unpause,
    compose_wait,
)
from .logs import compose_logs
from .pullpush import compose_pull, compose_push
from .runexec import compose_cp, compose_exec, compose_run
from .systemd import compose_systemd
from .updown import compose_down, compose_up
from .version import compose_version

__all__ = [
    "COMMAND_HANDLERS",
    "Handler",
    "install_handlers",
]

#: handler 统一签名：async def handler(compose, args) -> Any。
#: ``compose`` 为鸭子类型引擎（提供 podman/services/containers 等属性与
#: commands 查表），测试中以 stub/mock 注入。
Handler = Callable[[Any, argparse.Namespace], Awaitable[Any]]

#: 与上游 ``@cmd_run`` 注册名一一对应的显式注册表（24 项，TR-7.1 集合对等）。
COMMAND_HANDLERS: dict[str, Handler] = {
    "ls": list_running_projects,
    "version": compose_version,
    "wait": compose_wait,
    "systemd": compose_systemd,
    "pull": compose_pull,
    "push": compose_push,
    "build": compose_build,
    "up": compose_up,
    "down": compose_down,
    "ps": compose_ps,
    "run": compose_run,
    "cp": compose_cp,
    "exec": compose_exec,
    "start": compose_start,
    "stop": compose_stop,
    "restart": compose_restart,
    "logs": compose_logs,
    "config": compose_config,
    "port": compose_port,
    "pause": compose_pause,
    "unpause": compose_unpause,
    "kill": compose_kill,
    "stats": compose_stats,
    "images": compose_images,
}


def install_handlers(compose: Any) -> None:
    """把注册表灌入引擎的 ``commands`` 查表（替代装饰器副作用）。

    上游 handler 之间通过 ``compose.commands["up"]`` 等互相调用；
    T8 CLI 装配期调用本函数完成绑定，引擎实例化期仍保持零命令依赖。
    """
    compose.commands.update(COMMAND_HANDLERS)
