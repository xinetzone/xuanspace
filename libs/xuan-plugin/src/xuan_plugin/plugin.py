"""插件生命周期状态机 —— Cordis Fiber 实现。"""

import enum
import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .disposable import Disposable, DisposableList, DisposalError

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

# 摘要截断长度：16 位十六进制（64 bit），足以区分实现差异且便于日志阅读
_DIGEST_LEN = 16


def _digest(value: object) -> str:
    """计算实现对象的稳定摘要，作为 epoch 身份。

    不使用 ``id()``：对象地址跨进程不可比，会使同一装配在两次独立进程中
    得到不同的 epoch，从而无法复现装配结果。

    优先以「模块 + 限定名 + 源码摘要」构成结构身份——源码相同则身份相同，
    源码变更则身份变更，与对象地址无关。当源码不可得（内建对象、动态生成
    的实例）时回退到 ``repr`` 摘要：可区分同类型的不同实例，但不跨进程。
    """
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return f"{type(value).__name__}:{value!r}"

    module = getattr(value, "__module__", "") or ""
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    # getsource 的参数是联合类型，动态取值后静态类型无法收窄，故显式标注 Any
    target: Any
    if qualname:
        target = value
    else:
        cls = type(value)
        module = getattr(cls, "__module__", "") or module
        qualname = getattr(cls, "__qualname__", cls.__name__)
        target = cls

    try:
        source = inspect.getsource(target)
    except OSError, TypeError:
        source = ""
    if source:
        fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()[:_DIGEST_LEN]
    else:
        fingerprint = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:_DIGEST_LEN]
    return f"{module}:{qualname}:{fingerprint}"


def _config_digest(config: dict | None) -> str:
    """计算注入配置的稳定摘要，键序无关。

    ``sort_keys=True`` 保证 ``{"a": 1, "b": 2}`` 与 ``{"b": 2, "a": 1}``
    得到同一摘要，使 epoch 不因字典构造顺序而变化。
    """
    if not config:
        return "none"
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_LEN]


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
            errors: list[tuple[str, Exception]] = []
            for d in disposables:
                try:
                    d()
                except Exception as exc:  # noqa: BLE001 —— 聚合后统一上报，不中断同批撤销
                    errors.append((getattr(d, "__name__", "disposable"), exc))
            if errors:
                raise DisposalError(errors)

        return undo

    def _compute_epoch(self) -> str:
        """计算 epoch：依赖服务实现与注入配置的摘要拼接。

        无依赖时返回空字符串（总是激活）。与旧实现相比，``id(impl)`` 已替换为
        :func:`_digest` 摘要身份，并纳入 ``InjectSpec.config`` 的摘要——
        依赖实现或注入配置任一变化都会产生新 epoch，从而触发退役重插。
        """
        if not self.plugin.inject:
            return ""
        parts: list[str] = []
        for spec in self.plugin.inject:
            try:
                impl = self.ctx.get(spec.name)
            except KeyError:
                return _INACTIVE
            parts.append(f"{spec.name}:{_digest(impl)}:{_config_digest(spec.config)}")
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
        """执行插件的 apply 函数，加载服务。

        apply 中途失败时必须先撤销已登记的 effect，否则失败组件会留下
        部分装配痕迹，破坏「插入/退役是唯一外部输入」的隔离性。
        """
        self.state = FiberState.LOADING
        self._error = None
        try:
            self.plugin.apply(self.ctx, self.config)
            self.state = FiberState.ACTIVE
        except Exception as e:
            cleanup = self._disposables.dispose_all()
            self._disposables = DisposableList()
            self._error = DisposalError([("apply", e), *cleanup]) if cleanup else e
            self.state = FiberState.FAILED

    def _unload(self) -> None:
        """卸载插件：逆序执行所有 disposable，聚合异常且不中断。

        与旧实现的差异：不再用 ``contextlib.suppress`` 静默吞掉异常。全部
        撤销尝试完成后效应数必归零；若有失败则置 FAILED 并通过 ``error``
        暴露聚合异常。
        """
        self.state = FiberState.UNLOADING
        errors = self._disposables.dispose_all()
        self._disposables = DisposableList()
        if errors:
            self._error = DisposalError(errors)
            self.state = FiberState.FAILED
        else:
            self.state = FiberState.PENDING

    @property
    def error(self) -> Exception | None:
        """最近一次装配或卸载失败的异常；None 表示无错误。"""
        return self._error

    def dispose(self) -> None:
        """彻底销毁 Fiber。

        卸载期若发生异常，保留 FAILED 状态以便调用方通过 ``error`` 查证；
        否则置 DISPOSED。
        """
        if self.state in (FiberState.ACTIVE, FiberState.FAILED):
            self._unload()
        if self.state != FiberState.FAILED:
            self.state = FiberState.DISPOSED
