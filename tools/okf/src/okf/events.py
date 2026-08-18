"""事件系统 —— 五种分发模式。"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from .context import Context
from .disposable import Disposable


async def _await_if_needed(value: object) -> object:
    """等待可等待对象，否则原样返回。"""
    if isinstance(value, Awaitable):
        return await value
    return value


def _on(self: Context, event: str, handler: Callable) -> Disposable:
    """注册事件监听器，返回逆函数用于取消监听。"""
    handlers = self._listeners.setdefault(event, [])
    handlers.append(handler)

    removed = False

    def undo() -> None:
        nonlocal removed
        if not removed:
            with contextlib.suppress(ValueError):
                handlers.remove(handler)
            removed = True

    return undo


def _emit(self: Context, event: str, *args) -> None:
    """纯通知：同步执行所有监听器，无返回值。"""
    for handler in tuple(self._listeners.get(event, ())):
        handler(*args)


def _bail(self: Context, event: str, *args) -> object | None:
    """同步短路查找：第一个返回非 None 的结果获胜。"""
    for handler in tuple(self._listeners.get(event, ())):
        result = handler(*args)
        if result is not None:
            return result
    return None


def _parallel(self: Context, event: str, *args) -> list:
    """并行执行所有监听器（含异步），返回结果列表。"""

    async def run() -> list:
        return await asyncio.gather(
            *(_await_if_needed(handler(*args)) for handler in tuple(self._listeners.get(event, ())))
        )

    return asyncio.run(run())


def _serial(self: Context, event: str, *args) -> object | None:
    """串行执行：按注册顺序，第一个返回非 None 的结果获胜。"""
    for handler in tuple(self._listeners.get(event, ())):
        result = handler(*args)
        if result is not None:
            return result
    return None


def _waterfall(self: Context, event: str, *args) -> object:
    """中间件链：每个监听器接收 (*args, next)，调用 next() 继续。"""
    handlers = list(self._listeners.get(event, ()))
    index = 0

    def next_handler(*next_args) -> object:
        nonlocal index
        if index >= len(handlers):
            return next_args[0] if next_args else None
        handler = handlers[index]
        index += 1
        return handler(*next_args, next_handler)

    return next_handler(*args)


# 扩展 Context 类（monkey-patch）
Context.on = _on
Context.emit = _emit
Context.bail = _bail
Context.parallel = _parallel
Context.serial = _serial
Context.waterfall = _waterfall
