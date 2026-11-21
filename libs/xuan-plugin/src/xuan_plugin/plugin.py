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
    """Fiber 七态枚举（设计文档 §5.5）。

    ``INACTIVE`` 追加在末尾而非插入中间：既有六态的数值保持不变，避免外部
    按数值比对的状态快照（日志、文档表格）因新增状态整体漂移。
    """

    PENDING = 0
    LOADING = 1
    ACTIVE = 2
    UNLOADING = 3
    DISPOSED = 4
    FAILED = 5
    INACTIVE = 6


class FiberTransitionError(RuntimeError):
    """非法的生命周期迁移：目标状态不在 §5.5 迁移表内。

    这是编程错误而非运行期故障——例如对 ACTIVE 插件调用 ``retry()``，或在
    ``DISPOSED`` 终态上再次请求加载。快速失败以暴露调用方对状态机的误用，
    而不是静默吞掉。
    """

    def __init__(self, name: str, source: FiberState, target: FiberState) -> None:
        self.name = name
        self.source = source
        self.target = target
        super().__init__(f"插件 {name} 不允许从 {source.name} 迁移到 {target.name}")


# §5.5 迁移表的可执行形态。FAILED → INACTIVE / UNLOADING 不在图上的主路径中，
# 但分别对应「依赖消失」与「dispose 时回收失败节点」，二者都满足「无效应、原因可解释」。
_TRANSITIONS: dict[FiberState, frozenset[FiberState]] = {
    FiberState.PENDING: frozenset({FiberState.LOADING, FiberState.INACTIVE, FiberState.DISPOSED}),
    FiberState.INACTIVE: frozenset({FiberState.LOADING, FiberState.DISPOSED}),
    FiberState.LOADING: frozenset({FiberState.ACTIVE, FiberState.FAILED}),
    FiberState.ACTIVE: frozenset({FiberState.UNLOADING, FiberState.DISPOSED}),
    FiberState.UNLOADING: frozenset({FiberState.PENDING, FiberState.FAILED}),
    FiberState.FAILED: frozenset({FiberState.LOADING, FiberState.INACTIVE, FiberState.UNLOADING, FiberState.DISPOSED}),
    FiberState.DISPOSED: frozenset(),
}

_LIFECYCLES = frozenset({"reloadable", "static"})


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
    """插件定义。

    ``lifecycle`` 与清单的 K2 字段同口径：``reloadable`` 允许依赖变更时先卸后载，
    ``static`` 表示进程内只装配一次——原生 ``.so`` 无法真正卸载（ADR-03）。
    """

    name: str
    apply: Callable[[Context, dict], None]
    inject: list[InjectSpec] = field(default_factory=list)
    provide: list[str] = field(default_factory=list)
    lifecycle: str = "reloadable"

    def __post_init__(self) -> None:
        if self.lifecycle not in _LIFECYCLES:
            raise ValueError(f"未知生命周期 {self.lifecycle!r}，期望 {sorted(_LIFECYCLES)} 之一")


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
        self._missing: tuple[str, ...] = ()
        self._suppressed_reloads = 0
        self._inject_names: frozenset[str] = frozenset(s.name for s in plugin.inject)

    @property
    def name(self) -> str:
        return self.plugin.name

    @property
    def inject_names(self) -> frozenset[str]:
        """缓存的依赖服务名集合，避免每次 notify 重复计算。"""
        return self._inject_names

    @property
    def reason(self) -> tuple[str, ...]:
        """INACTIVE 的失活原因（未注册的必需注入）；其它状态恒为空元组。

        对应 §5.5 的不变量「INACTIVE 未持有 effect，且原因可解释」——调用方无需
        猜「为什么没生效」，直接读缺失键即可（V4）。
        """
        return self._missing if self.state is FiberState.INACTIVE else ()

    @property
    def suppressed_reloads(self) -> int:
        """静态插件被抑制的热换次数。

        ``lifecycle="static"`` 的插件在 ACTIVE 后不再响应依赖变更（ADR-03）；
        该计数让「抑制」可观测，而不是静默忽略。
        """
        return self._suppressed_reloads

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

    def _transition(self, target: FiberState) -> None:
        """按 §5.5 迁移表切换状态；表外迁移抛 :class:`FiberTransitionError`。"""
        if target not in _TRANSITIONS[self.state]:
            raise FiberTransitionError(self.plugin.name, self.state, target)
        self.state = target

    def _missing_keys(self) -> tuple[str, ...]:
        """当前未注册的必需注入名（用于 INACTIVE 的原因）。"""
        missing: list[str] = []
        for spec in self.plugin.inject:
            try:
                self.ctx.get(spec.name)
            except KeyError:
                missing.append(spec.name)
        return tuple(missing)

    def _refresh(self) -> None:
        """重新计算 epoch，并按 §5.5 迁移到 INACTIVE / LOADING / UNLOADING。"""
        if self.state is FiberState.DISPOSED:
            raise FiberTransitionError(self.plugin.name, self.state, FiberState.LOADING)
        if self.state in (FiberState.LOADING, FiberState.UNLOADING):
            # apply/撤销内部的 provide 会触发 notify 重入；迁移进行中不重算，
            # 否则会得到 LOADING → LOADING 这类表外迁移。
            return

        new_epoch = self._compute_epoch()
        missing = self._missing_keys() if new_epoch == _INACTIVE else ()

        # ADR-03：静态插件 ACTIVE 后不热换。直接抑制而非抛错，避免依赖方一次
        # 普通的 provide 撤销就把原生插件踢进非法迁移。
        if self.plugin.lifecycle == "static" and self.state is FiberState.ACTIVE and new_epoch != self._epoch:
            self._suppressed_reloads += 1
            return

        if new_epoch == self._epoch:
            if new_epoch == _INACTIVE:
                # 初次刷新即不可解（PENDING）、或原因集合变化（INACTIVE）：
                # 都归一到 _deactivate，避免停在「未尝试」的 PENDING 上。
                self._deactivate(missing)
            return

        self._epoch = new_epoch

        if new_epoch == _INACTIVE:
            self._deactivate(missing)
            return

        if self.state is FiberState.ACTIVE:
            self._unload()  # 依赖变更：先卸旧实现
        self._reload()  # 再从 PENDING / INACTIVE / FAILED 载入新实现

    def _deactivate(self, missing: tuple[str, ...]) -> None:
        """进入 INACTIVE：先确保无残留效应，再记录原因。"""
        if self.state is FiberState.ACTIVE:
            self._unload()
            if self.state is FiberState.FAILED:
                return  # 卸载失败保持 FAILED，避免把 DisposalError 掩盖成「待激活」
        if self.state is FiberState.INACTIVE:
            self._missing = missing
            return
        self._error = None
        self._missing = missing
        self._transition(FiberState.INACTIVE)

    def _reload(self) -> None:
        """执行插件的 apply 函数，加载服务。

        apply 中途失败时必须先撤销已登记的 effect，否则失败组件会留下
        部分装配痕迹，破坏「插入/退役是唯一外部输入」的隔离性。
        """
        self._transition(FiberState.LOADING)
        self._error = None
        try:
            self.plugin.apply(self.ctx, self.config)
            self._missing = ()
            self._transition(FiberState.ACTIVE)
        except Exception as e:
            cleanup = self._disposables.dispose_all()
            self._disposables = DisposableList()
            self._error = DisposalError([("apply", e), *cleanup]) if cleanup else e
            self._transition(FiberState.FAILED)

    def _unload(self) -> None:
        """卸载插件：逆序执行所有 disposable，聚合异常且不中断。

        与旧实现的差异：不再用 ``contextlib.suppress`` 静默吞掉异常。全部
        撤销尝试完成后效应数必归零；若有失败则置 FAILED 并通过 ``error``
        暴露聚合异常。
        """
        self._transition(FiberState.UNLOADING)
        errors = self._disposables.dispose_all()
        self._disposables = DisposableList()
        if errors:
            self._error = DisposalError(errors)
            self._transition(FiberState.FAILED)
        else:
            self._missing = ()
            self._transition(FiberState.PENDING)

    def retry(self) -> None:
        """显式重试（§5.5 FAILED → LOADING）。

        只对 ``FAILED`` 开放：ACTIVE 无需重试，INACTIVE 的迁移由依赖出现自动
        触发。若依赖仍不可得，重试退回 INACTIVE 而不是再次失败——这与
        「依赖不可得不是异常」的解析语义一致。
        """
        if self.state is not FiberState.FAILED:
            raise FiberTransitionError(self.plugin.name, self.state, FiberState.LOADING)

        new_epoch = self._compute_epoch()
        self._epoch = new_epoch
        self._error = None
        if new_epoch == _INACTIVE:
            self._missing = self._missing_keys()
            self._transition(FiberState.INACTIVE)
            return
        self._reload()

    @property
    def error(self) -> Exception | None:
        """最近一次装配或卸载失败的异常；None 表示无错误。"""
        return self._error

    def dispose(self) -> None:
        """彻底销毁 Fiber；``DISPOSED`` 为终态，重复调用幂等。

        卸载期若发生异常，保留 FAILED 状态以便调用方通过 ``error`` 查证；
        否则迁移到 DISPOSED。静态插件在此仍执行**逻辑退役**——撤销能力句柄
        与效应，但不承诺已映射 ``.so`` 的机器码被卸载（ADR-03）。
        """
        if self.state is FiberState.DISPOSED:
            return
        if self.state in (FiberState.ACTIVE, FiberState.FAILED):
            self._unload()
        if self.state is not FiberState.FAILED:
            self._transition(FiberState.DISPOSED)
