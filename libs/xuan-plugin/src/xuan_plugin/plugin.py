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
        self._inputs: tuple[tuple[str, str, str], ...] | None = None
        self._pending: str | None = None
        self._reloads = 0
        self._rollbacks = 0
        self._last_reload_error: Exception | None = None
        self._reload_reason: tuple[str, ...] = ()

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

    @property
    def reloads(self) -> int:
        """成功完成的热重载次数（不含首次装配）。"""
        return self._reloads

    @property
    def rollbacks(self) -> int:
        """热重载失败并回滚已登记效应的次数。

        「回滚」的边界是**不留半成品**：撤销本次已登记的效应、效应数归零，
        而不是恢复被覆盖的依赖实现或外部副作用——后者在 Python 进程内不可逆
        （ADR-03），故以 FAILED + 原始异常显式暴露（V1）。
        """
        return self._rollbacks

    @property
    def last_reload_error(self) -> Exception | None:
        """最近一次热重载失败的原因；成功重载后保留，直到下一次失败覆盖。"""
        return self._last_reload_error

    @property
    def reload_reason(self) -> tuple[str, ...]:
        """最近一次热重载的逐字段变化，形如 ``dep:impl`` / ``dep:config``。

        对应 §5.4 第 7 步的「逐字段对账」：只有摘要变化的依赖字段才会出现在
        这里，未变化的依赖不构成重载理由（同一 epoch 不会触发重载）。
        """
        return self._reload_reason

    @property
    def pending_reload(self) -> str | None:
        """失败重载对应的目标 epoch；无待重试目标时为 None。"""
        return self._pending

    def effect(self, execute: Callable[[], Disposable | list[Disposable]], label: str = "") -> Disposable:
        """注册可逆效应，返回逆函数。"""
        disposables = execute()
        if callable(disposables) and not isinstance(disposables, list):
            disposables = [disposables]

        removers = [self._disposables.push(d) for d in disposables]

        def undo() -> None:
            errors: list[tuple[str, Exception]] = []
            for remove, d in zip(removers, disposables):
                remove()
                try:
                    d()
                except Exception as exc:  # noqa: BLE001 —— 聚合后统一上报，不中断同批撤销
                    errors.append((getattr(d, "__name__", "disposable"), exc))
            if errors:
                raise DisposalError(errors)

        return undo

    def _snapshot_inputs(self) -> tuple[tuple[str, str, str], ...] | None:
        """采集依赖身份快照：``(服务名, 实现摘要, 配置摘要)``。

        任一必需依赖缺失时返回 None（对应 epoch ``_INACTIVE``）。快照既用于
        计算 epoch，也用于重载时逐字段对账「到底哪一个字段变了」。
        """
        if not self.plugin.inject:
            return ()
        rows: list[tuple[str, str, str]] = []
        for spec in self.plugin.inject:
            try:
                impl = self.ctx.get(spec.name)
            except KeyError:
                return None
            rows.append((spec.name, _digest(impl), _config_digest(spec.config)))
        return tuple(rows)

    @staticmethod
    def _epoch_of(inputs: tuple[tuple[str, str, str], ...] | None) -> str:
        """把依赖快照编码为 epoch 字符串；None 编码为 ``_INACTIVE``。"""
        if inputs is None:
            return _INACTIVE
        if not inputs:
            return ""
        return "|".join(f"{name}:{impl}:{config}" for name, impl, config in inputs)

    def _compute_epoch(self) -> str:
        """计算 epoch：依赖服务实现与注入配置的摘要拼接。

        无依赖时返回空字符串（总是激活）。与旧实现相比，``id(impl)`` 已替换为
        :func:`_digest` 摘要身份，并纳入 ``InjectSpec.config`` 的摘要——
        依赖实现或注入配置任一变化都会产生新 epoch，从而触发退役重插。
        """
        return self._epoch_of(self._snapshot_inputs())

    @staticmethod
    def _describe_change(
        old: tuple[tuple[str, str, str], ...] | None,
        new: tuple[tuple[str, str, str], ...] | None,
    ) -> tuple[str, ...]:
        """对比两份依赖快照，返回变化的字段标签（字典序）。"""
        if old is None or new is None:
            return ()
        old_map = {name: (impl, config) for name, impl, config in old}
        new_map = {name: (impl, config) for name, impl, config in new}
        changes: list[str] = []
        for name, (impl, config) in new_map.items():
            previous = old_map.get(name)
            if previous is None:
                changes.append(f"{name}:added")
                continue
            if previous[0] != impl:
                changes.append(f"{name}:impl")
            if previous[1] != config:
                changes.append(f"{name}:config")
        changes.extend(f"{name}:lost" for name in old_map.keys() - new_map.keys())
        return tuple(sorted(changes))

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

        inputs = self._snapshot_inputs()
        new_epoch = self._epoch_of(inputs)
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
            elif self.state is FiberState.ACTIVE and self._pending is not None:
                # 依赖已回到当前生效身份，此前的失败重载不再有待办意义。
                self._pending = None
            return

        if new_epoch == _INACTIVE:
            self._pending = None
            self._deactivate(missing)
            return

        if self.state is FiberState.ACTIVE:
            self._replace(inputs, new_epoch)
        else:
            self._epoch = new_epoch
            self._reload()

    def _replace(
        self,
        inputs: tuple[tuple[str, str, str], ...] | None,
        new_epoch: str,
    ) -> None:
        """热替换事务：先撤旧效应再应用新实现，失败则回滚到零残留。

        「备份」体现为 ``old_inputs`` 快照与 ``reload_reason`` 的逐字段对账；
        「恢复」的边界是不留半成品——新实现 ``apply`` 失败时撤销其已登记效应，
        旧实现因依赖已被覆盖而无法原地还原（Python 进程内不可逆，V1），
        故以 FAILED + 原始异常显式暴露，由 ``retry()`` 决定是否再次尝试。
        """
        reason = self._describe_change(self._inputs, inputs)
        self._unload()
        if self.state is FiberState.FAILED:
            # 旧效应的撤销本身失败：保留 FAILED 与聚合异常，不再尝试新实现。
            self._pending = new_epoch
            self._reload_reason = reason
            return

        self._epoch = new_epoch
        self._reload()
        if self.state is FiberState.ACTIVE:
            self._reloads += 1
            self._reload_reason = reason
            return

        # 新实现失败：已登记效应由上方的 _reload 回滚为零，记录待重试目标。
        self._rollbacks += 1
        self._last_reload_error = self._error
        self._pending = new_epoch
        self._reload_reason = reason

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
        self._pending = None
        self._reload_reason = ()
        self._transition(FiberState.INACTIVE)

    def _reload(self) -> None:
        """执行插件的 apply 函数，加载服务。

        apply 中途失败时必须先撤销已登记的 effect，否则失败组件会留下
        部分装配痕迹，破坏「插入/退役是唯一外部输入」的隔离性。``apply``
        期间经 ``Context.effect`` 登记的效应归属本 Fiber（``_apply_scope``），
        使「撤销」在热替换时真正发生，而不是留在根上下文里。
        """
        self._transition(FiberState.LOADING)
        self._error = None
        try:
            with self.ctx._apply_scope(self):
                self.plugin.apply(self.ctx, self.config)
            self._missing = ()
            self._inputs = self._snapshot_inputs()
            self._pending = None
            self._transition(FiberState.ACTIVE)
        except Exception as e:
            cleanup = self._disposables.dispose_all()
            self._disposables = DisposableList()
            self._inputs = None
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
            self._inputs = None
            self._transition(FiberState.FAILED)
        else:
            self._missing = ()
            self._inputs = None
            self._transition(FiberState.PENDING)

    def retry(self) -> None:
        """显式重试（§5.5 FAILED → LOADING）。

        只对 ``FAILED`` 开放：ACTIVE 无需重试，INACTIVE 的迁移由依赖出现自动
        触发。若依赖仍不可得，重试退回 INACTIVE 而不是再次失败——这与
        「依赖不可得不是异常」的解析语义一致。重试成功计入 ``reloads``，
        使「失败—重试—恢复」在观测上与自动热重载同口径。
        """
        if self.state is not FiberState.FAILED:
            raise FiberTransitionError(self.plugin.name, self.state, FiberState.LOADING)

        self._pending = None
        new_epoch = self._compute_epoch()
        self._epoch = new_epoch
        self._error = None
        if new_epoch == _INACTIVE:
            self._missing = self._missing_keys()
            self._transition(FiberState.INACTIVE)
            return
        self._reload()
        if self.state is FiberState.ACTIVE:
            self._reloads += 1

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
            self._pending = None
            self._transition(FiberState.DISPOSED)
