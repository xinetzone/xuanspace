"""统一上下文 —— Cordis Context 实现。"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .disposable import Disposable, DisposableList, EffectMeta
from .plugin import Fiber, FiberState, Plugin


async def _await_if_needed(value: object) -> object:
    """等待可等待对象，否则原样返回。"""
    if isinstance(value, Awaitable):
        return await value
    return value


@dataclass
class Context:
    """统一上下文，同时承载效应追踪、服务注册与插件装配。

    对应 Cordis 的 Context——"统一上下文类型"。
    """

    _store: dict[str, object] = field(default_factory=dict)
    _fibers: dict[str, Fiber] = field(default_factory=dict)
    _effects: list[EffectMeta] = field(default_factory=list)
    _root_disposables: DisposableList = field(default_factory=DisposableList)
    _listeners: dict[str, list[Callable]] = field(default_factory=dict)

    def effect(self, execute: Callable[[], Disposable | list[Disposable]], label: str = "") -> Disposable:
        """注册可逆效应到根 DisposableList。"""
        disposables = execute()
        if callable(disposables) and not isinstance(disposables, list):
            disposables = [disposables]

        for d in disposables:
            self._root_disposables.push(d)

        if label:
            self._effects.append(EffectMeta(label=label))

        def undo() -> None:
            for d in disposables:
                with contextlib.suppress(Exception):
                    d()

        return undo

    def provide(self, name: str, impl: object) -> Disposable:
        """供应服务实现，返回逆函数（撤销供应）。

        供应后自动调用 notify 通知所有依赖该服务的插件。
        """
        self._store[name] = impl

        def undo() -> None:
            if self._store.get(name) is impl:
                del self._store[name]
                self.notify([name])

        self.notify([name])
        return undo

    def get(self, name: str) -> object:
        """获取服务实现。KeyError 表示未注册。"""
        if name not in self._store:
            raise KeyError(f"Service '{name}' not registered")
        return self._store[name]

    def plugin(self, plugin: Plugin, config: dict | None = None) -> None:
        """装配插件，创建 Fiber 并启动加载。"""
        fiber = Fiber(plugin, self, config)
        self._fibers[plugin.name] = fiber
        # 计算初始 epoch 并尝试激活
        fiber._refresh()

    def notify(self, names: list[str]) -> None:
        """响应式通知：遍历所有 Fiber，对 inject 命中者执行 _refresh。"""
        name_set = set(names)
        for fiber in self._fibers.values():
            if fiber.state == FiberState.DISPOSED:
                continue
            inject_names = fiber.inject_names
            if not inject_names:
                continue  # 无依赖的插件不受通知影响
            if inject_names & name_set:
                fiber._refresh()

    def get_effects(self) -> list[EffectMeta]:
        """获取效应元信息树（用于调试与可观测性）。"""
        return list(self._effects)

    def dispose(self) -> None:
        """清理所有资源：逆序卸载所有 Fiber 并清空服务。"""
        for fiber in reversed(list(self._fibers.values())):
            with contextlib.suppress(Exception):
                fiber.dispose()
        self._fibers.clear()
        for dispose in self._root_disposables.clear():
            with contextlib.suppress(Exception):
                dispose()
        self._store.clear()
        self._effects.clear()

    # ─── 上下文管理器协议 ───────────────────────────────────────────────

    def __enter__(self) -> Context:
        """进入上下文：资源回收由 ``with`` 保证（contextlib 语义落地）。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文：自动 ``dispose()``，无论代码块是否抛异常。"""
        self.dispose()

    # ─── 事件分发（on/emit/bail/parallel/serial/waterfall） ─────────────

    def on(self, event: str, handler: Callable) -> Disposable:
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

    def emit(self, event: str, *args) -> None:
        """纯通知：同步执行所有监听器，无返回值。"""
        for handler in tuple(self._listeners.get(event, ())):
            handler(*args)

    def bail(self, event: str, *args) -> object | None:
        """同步短路查找：第一个返回非 None 的结果获胜。"""
        for handler in tuple(self._listeners.get(event, ())):
            result = handler(*args)
            if result is not None:
                return result
        return None

    def parallel(self, event: str, *args) -> list:
        """并行执行所有监听器（含异步），返回结果列表。"""

        async def run() -> list:
            return await asyncio.gather(
                *(_await_if_needed(handler(*args)) for handler in tuple(self._listeners.get(event, ())))
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run())
        raise RuntimeError(
            "Cannot call Context.parallel() synchronously from within a running event loop; "
            "use asyncio.run(Context.parallel(...)) outside the loop or await listeners directly."
        )

    def serial(self, event: str, *args) -> object | None:
        """串行执行：按注册顺序，第一个返回非 None 的结果获胜。"""
        for handler in tuple(self._listeners.get(event, ())):
            result = handler(*args)
            if result is not None:
                return result
        return None

    def waterfall(self, event: str, *args) -> object:
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
