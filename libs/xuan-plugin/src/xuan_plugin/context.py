"""统一上下文 —— Cordis Context 实现。"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .disposable import Disposable, DisposableList, DisposalError, EffectMeta
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
    _disposal_errors: list[tuple[str, Exception]] = field(default_factory=list)

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
            errors: list[tuple[str, Exception]] = []
            for d in disposables:
                try:
                    d()
                except Exception as exc:  # noqa: BLE001 —— 聚合后统一上报，不中断同批撤销
                    errors.append((getattr(d, "__name__", "disposable"), exc))
            if errors:
                raise DisposalError(errors)

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

    def registered_keys(self) -> tuple[str, ...]:
        """已注册的能力键（字典序），用于对账与回滚后的残留断言。"""
        return tuple(sorted(self._store))

    def plugin(self, plugin: Plugin, config: dict | None = None) -> Fiber:
        """装配插件，创建 Fiber 并启动加载，返回该 Fiber 供调用方查证状态。"""
        fiber = Fiber(plugin, self, config)
        self._fibers[plugin.name] = fiber
        # 计算初始 epoch 并尝试激活
        fiber._refresh()
        return fiber

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

    def dispose(self, report: Callable[[str, Exception], None] | None = None) -> list[tuple[str, Exception]]:
        """清理所有资源：逆序卸载 Fiber 并回收根效应。

        卸载期异常不中断流程——所有 Fiber 与根效应都会被执行，失败项汇总
        返回并留存于 ``get_disposal_errors()``。``report`` 可选，用于逐项
        即时上报（如写日志）。不主动抛出：``__exit__`` 中抛异常会掩盖
        代码块原本的异常。
        """
        errors: list[tuple[str, Exception]] = []
        for fiber in reversed(list(self._fibers.values())):
            fiber.dispose()
            if fiber.error is not None:
                label = f"fiber:{fiber.name}"
                errors.append((label, fiber.error))
                if report is not None:
                    report(label, fiber.error)
        self._fibers.clear()
        for label, exc in self._root_disposables.dispose_all(report):
            errors.append((f"root:{label}", exc))
        self._store.clear()
        self._effects.clear()
        self._disposal_errors = errors
        return errors

    def get_disposal_errors(self) -> list[tuple[str, Exception]]:
        """返回上一次 ``dispose()`` 期间的失败清单。"""
        return list(self._disposal_errors)

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
