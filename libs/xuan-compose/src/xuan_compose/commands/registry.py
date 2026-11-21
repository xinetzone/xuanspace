# SPDX-License-Identifier: GPL-2.0-only
"""命令登记内核（P2-2）——把静态处理器表改造为可撤销登记。

上游 ``@cmd_run`` 装饰器在类构造期把 handler 挂到 ``self.commands``；
分层后的实现用三张模块级静态表（handler / parser / help）表达同一事实，
登记即副作用、撤销无证据。本模块把「一条指令」收敛为单个
:class:`CommandSpec`，并给出可撤销的登记语义：

1. :meth:`CommandRegistry.register` 返回 :data:`xuan_plugin.Disposable`，
   调用逆函数即摘除该条登记；
2. :meth:`CommandRegistry.register_all` 用 ``DisposableList`` 栈式逆序回滚，
   中途冲突不留下半装配状态；
3. :meth:`CommandRegistry.bind` 把 handler 灌入引擎查表并返回逆函数，
   撤销时逐项恢复绑定前取值（原本不存在的键被删除）。

本模块只提供容器语义，不持有任何具体指令；唯一事实源是
:data:`xuan_compose.cli.command_specs.COMMAND_SPECS`。
"""

import argparse
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from xuan_plugin import Disposable, DisposableList

__all__ = [
    "CommandRegistry",
    "CommandSpec",
    "Handler",
    "ParserFn",
]

#: handler 统一签名：async def handler(compose, args) -> Any。
#: ``compose`` 为鸭子类型引擎（提供 podman/services/containers 等属性与
#: commands 查表），测试中以 stub/mock 注入。
Handler = Callable[[Any, argparse.Namespace], Awaitable[Any]]

#: parser 挂载函数统一签名：fn(subparser) -> None。
ParserFn = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """一条指令的完整声明：名字、handler、parser 挂载序、help 文本。"""

    name: str
    handler: Handler
    parsers: tuple[ParserFn, ...] = ()
    help: str = ""
    description: str | None = None


class CommandRegistry:
    """按登记序保存 ``CommandSpec`` 的最小注册表。

    登记序即 subparser 的创建序（与上游 ``@cmd_run`` 出现序一致）。
    """

    def __init__(self) -> None:
        self._specs: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> Disposable:
        """登记一条指令，返回撤销函数（重复登记同一名字视为契约冲突）。"""
        if spec.name in self._specs:
            raise ValueError(f"命令重复登记: {spec.name}")
        self._specs[spec.name] = spec

        def undo() -> None:
            if self._specs.get(spec.name) is spec:
                del self._specs[spec.name]

        return undo

    def register_all(self, specs: Iterable[CommandSpec]) -> DisposableList:
        """批量登记；任一条失败即逆序回滚已成功的部分再抛出。"""
        batch = DisposableList()
        try:
            for spec in specs:
                batch.push(self.register(spec))
        except Exception:
            batch.dispose_all()
            raise
        return batch

    def bind(self, compose: Any) -> Disposable:
        """把 handler 灌入 ``compose.commands``，返回恢复绑定前状态的逆函数。"""
        target: dict[str, Any] = compose.commands
        previous = {name: target[name] for name in self._specs if name in target}
        target.update({name: spec.handler for name, spec in self._specs.items()})
        done = False

        def unbind() -> None:
            nonlocal done
            if done:
                return
            done = True
            for name, spec in self._specs.items():
                # 只回收仍指向本登记 handler 的键，避免撤销覆盖后续登记
                if target.get(name) is not spec.handler:
                    continue
                if name in previous:
                    target[name] = previous[name]
                else:
                    target.pop(name, None)

        return unbind

    def __iter__(self) -> Iterator[CommandSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    @property
    def specs(self) -> MappingProxyType[str, CommandSpec]:
        """只读登记视图（键为命令名，迭代序为登记序）。"""
        return MappingProxyType(self._specs)
