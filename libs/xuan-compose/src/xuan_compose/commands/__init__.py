# SPDX-License-Identifier: GPL-2.0-only
"""命令层（T7）——24 个 handler 的实现与引擎绑定入口。

上游 podman_compose.py（第 3280-3316 行）用 ``@cmd_run`` 装饰器在类构造期
把 handler 挂到 ``self.commands``；分层后 handler 按上游行段切分到
``build/inspect/lifecycle/logs/pullpush/runexec/systemd/updown/version``
九个模块：

- ``version``：3378-3388；``systemd``：3411-3513；
- ``pullpush``：3516-3546；``build``：3688-3764；
- ``updown``：802-835、3767-3942、4109-4526；
- ``lifecycle``：3400-3408、4708-4755、4817-4868；
- ``runexec``：4549-4681；``inspect``：3324-3375、4530-4541、4795-4935；
- ``logs``：4759-4792。

P2-2 起，指令声明（handler、parser 挂载序、help）收敛为单一清单
:data:`xuan_compose.cli.command_specs.COMMAND_SPECS`，本包不再持有处理器表，
只保留 :func:`install_handlers` 这一引擎侧绑定入口；登记返回
``Disposable``，撤销即恢复绑定前的 ``compose.commands``。
"""

from typing import Any

from xuan_plugin import Disposable

from .registry import CommandRegistry, CommandSpec, Handler, ParserFn

__all__ = [
    "CommandRegistry",
    "CommandSpec",
    "Handler",
    "ParserFn",
    "install_handlers",
]


def install_handlers(compose: Any, registry: CommandRegistry | None = None) -> Disposable:
    """把已登记的 handler 灌入引擎 ``commands`` 查表。

    上游 handler 之间通过 ``compose.commands["up"]`` 等互相调用；
    T8 CLI 装配期调用本函数完成绑定，引擎实例化期仍保持零命令依赖。
    ``registry`` 省略时按唯一清单
    :func:`~xuan_compose.cli.command_specs.build_registry` 临时装配。
    """
    if registry is None:
        from ..cli.command_specs import build_registry

        registry = build_registry()
    return registry.bind(compose)
