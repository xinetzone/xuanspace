"""插件生命周期状态机 —— Cordis Fiber 实现。"""

import contextlib
import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .disposable import Disposable, DisposableList

if TYPE_CHECKING:
    from .context import Context


class FiberState(enum.IntEnum):
    """Fiber 六态枚举。"""
    PENDING = 0
    LOADING = 1
    ACTIVE = 2
    UNLOADING = 3
    DISPOSED = 4
    FAILED = 5


# 哨兵值：表示依赖未满足
_INACTIVE = "__INACTIVE__"


@dataclass(frozen=True, slots=True)
class InjectSpec:
    """依赖声明：插件需要哪些服务。"""
    name: str
    config: dict | None = None


@dataclass(frozen=True, slots=True)
class Plugin:
    """插件定义。"""
    name: str
    apply: Callable[[Context, dict], None]
    inject: list[InjectSpec] = field(default_factory=list)
    provide: list[str] = field(default_factory=list)


class Fiber:
    """插件运行时实例，管理单个插件的生命周期。"""

    def __init__(self, plugin: Plugin, ctx: Context, config: dict | None = None) -> None:
        self.plugin = plugin
        self.ctx = ctx
        self.config = config or {}
        self.state: FiberState = FiberState.PENDING
        self._disposables = DisposableList()
        self._error: Exception | None = None
        self._epoch: str = _INACTIVE
        self._inject_names: frozenset[str] = frozenset(s.name for s in plugin.inject)

    @property
    def name(self) -> str:
        return self.plugin.name

    @property
    def inject_names(self) -> frozenset[str]:
        """缓存的依赖服务名集合，避免每次 notify 重复计算。"""
        return self._inject_names

    def effect(self, execute: Callable[[], Disposable | list[Disposable]], label: str = "") -> Disposable:
        """注册可逆效应，返回逆函数。"""
        disposables = execute()
        if callable(disposables) and not isinstance(disposables, list):
            disposables = [disposables]

        for d in disposables:
            self._disposables.push(d)

        def undo() -> None:
            for d in disposables:
                with contextlib.suppress(Exception):
                    d()

        return undo

    def _compute_epoch(self) -> str:
        """计算 epoch：所有依赖服务实现的唯一标识拼接。

        如果没有依赖，返回空字符串（总是激活）。
        """
        if not self.plugin.inject:
            return ""
        parts: list[str] = []
        for spec in self.plugin.inject:
            try:
                impl = self.ctx.get(spec.name)
                parts.append(f"{spec.name}:{id(impl)}")
            except KeyError:
                return _INACTIVE
        return "|".join(parts)

    def _refresh(self) -> None:
        """重新计算 epoch 并触发激活/停用/中性。"""
        new_epoch = self._compute_epoch()
        if new_epoch == self._epoch:
            return  # 中性：epoch 不变，保持当前状态

        old_epoch = self._epoch
        self._epoch = new_epoch

        if old_epoch == _INACTIVE and new_epoch != _INACTIVE:
            self._reload()  # 依赖从无到有 → 激活
        elif old_epoch != _INACTIVE and new_epoch == _INACTIVE:
            self._unload()  # 依赖从有到无 → 停用
        elif old_epoch != _INACTIVE and new_epoch != _INACTIVE:
            self._unload()  # 依赖变更：先卸载旧实现
            self._reload()  # 再加载新实现

    def _reload(self) -> None:
        """执行插件的 apply 函数，加载服务。"""
        self.state = FiberState.LOADING
        self._error = None
        try:
            self.plugin.apply(self.ctx, self.config)
            self.state = FiberState.ACTIVE
        except Exception as e:
            self._error = e
            self.state = FiberState.FAILED

    def _unload(self) -> None:
        """卸载插件：逆序执行所有 disposable。"""
        self.state = FiberState.UNLOADING
        for dispose in self._disposables.clear():
            with contextlib.suppress(Exception):
                dispose()
        self._disposables = DisposableList()
        self.state = FiberState.PENDING

    def dispose(self) -> None:
        """彻底销毁 Fiber。"""
        if self.state in (FiberState.ACTIVE, FiberState.FAILED):
            self._unload()
        self.state = FiberState.DISPOSED
