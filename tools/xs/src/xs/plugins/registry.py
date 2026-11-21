"""命令注册表：插件以可撤销方式把命令汇聚到根 Typer 应用。

注册表只保存「名字 → 目标」的登记项，应用构建在装配完成后一次性进行。
这样撤销登记即可回到装配前的空状态，不必依赖 Click/Typer 未公开的卸载接口。
"""

from collections.abc import Callable
from dataclasses import dataclass

import typer
from xuan_plugin.disposable import Disposable

CommandTarget = typer.Typer | Callable[..., object]


@dataclass(frozen=True, slots=True)
class CommandEntry:
    """一条命令登记项。``kind`` 取 ``command`` 或 ``group``。"""

    name: str
    target: CommandTarget
    kind: str = "command"


class CommandRegistry:
    """保持登记顺序的命令表；``register`` 返回撤销函数。"""

    def __init__(self) -> None:
        self._entries: dict[str, CommandEntry] = {}

    def register(self, entry: CommandEntry) -> Disposable:
        if entry.name in self._entries:
            raise ValueError(f"命令重复登记：{entry.name}")
        self._entries[entry.name] = entry
        removed = False

        def undo() -> None:
            nonlocal removed
            if not removed and entry.name in self._entries:
                del self._entries[entry.name]
                removed = True

        return undo

    def entries(self) -> tuple[CommandEntry, ...]:
        """按登记顺序返回全部条目。"""
        return tuple(self._entries.values())
