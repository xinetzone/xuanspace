"""统一上下文 —— Cordis Context 实现。"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

from .disposable import Disposable, DisposableList, EffectMeta
from .plugin import Fiber, FiberState, Plugin


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
        for fiber in self._fibers.values():
            if fiber.state == FiberState.DISPOSED:
                continue
            # 检查该 fiber 的依赖是否与变更的服务名有交集
            inject_names = {s.name for s in fiber.plugin.inject}
            if not inject_names:
                continue  # 无依赖的插件不受通知影响
            if inject_names & set(names):
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
