"""统一上下文 —— Cordis Context 实现。"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field

from .disposable import Disposable, DisposableList, DisposalError, EffectMeta
from .plugin import Fiber, FiberState, Plugin

# 单次排空的轮次上限：正常级联在 2–3 轮内收敛（提供方 → 依赖方 → 更下游）。
# 上限用于把「互相提供且摘要每轮都变」的病态环转成可诊断错误，而不是挂死进程。
_MAX_FLUSH_ROUNDS = 100


class PluginNameConflictError(RuntimeError):
    """同名插件重复装配：已有 Fiber 仍在生命周期内（P2-11）。

    直接覆写 ``_fibers`` 会让旧实例的供应与效应失去持有者——``dispose`` 按字典
    遍历，旧 Fiber 不会被回收，键也留在 ``_store`` 里成为孤儿。故默认拒绝覆写，
    要求调用方先显式 ``dispose`` 旧实例；已 ``DISPOSED`` 的名称可重新占用。
    """

    def __init__(self, name: str, state: FiberState) -> None:
        self.name = name
        self.state = state
        super().__init__(
            f"插件 {name} 已装配且状态为 {state.name}，拒绝同名覆写；"
            f"请先 dispose 旧 Fiber 再重新装配，避免其供应与效应成为孤儿"
        )


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
    _apply_stack: list[Fiber] = field(default_factory=list)
    _batch_depth: int = 0
    _deferred: set[str] = field(default_factory=set)

    @contextlib.contextmanager
    def _apply_scope(self, fiber: Fiber) -> Iterator[None]:
        """进入某 Fiber 的 ``apply`` 作用域：期间登记的效应归属该 Fiber。

        归属决定「谁负责撤销」：应用期效应随 Fiber 卸载而逆序回收，否则热替换
        只换了代码、旧服务的注册残留仍在，``provide`` 撤销也不会发生。用栈而非
        单值，覆盖「apply 内提供能力 → notify → 依赖方就地装配」的嵌套场景。
        """
        self._apply_stack.append(fiber)
        try:
            yield
        finally:
            self._apply_stack.pop()

    def effect(self, execute: Callable[[], Disposable | list[Disposable]], label: str = "") -> Disposable:
        """注册可逆效应，返回逆函数。

        处于 ``apply`` 作用域时归属当前 Fiber，否则归属根 ``DisposableList``。
        返回的逆函数同时把效应从句柄列表中摘除，避免「手动撤销后再被卸载重放」
        导致同一逆操作执行两次。
        """
        disposables = execute()
        if callable(disposables) and not isinstance(disposables, list):
            disposables = [disposables]

        target = self._apply_stack[-1]._disposables if self._apply_stack else self._root_disposables
        removers = [target.push(d) for d in disposables]

        if label:
            self._effects.append(EffectMeta(label=label))

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
        """装配插件，创建 Fiber 并启动加载，返回该 Fiber 供调用方查证状态。

        同名插件只允许在旧实例已 ``DISPOSED`` 后重新装配；其余状态一律抛
        :class:`PluginNameConflictError`，不做静默覆写（否则旧实例的供应与效应
        会在 ``dispose`` 遍历中缺席，成为孤儿）。
        """
        existing = self._fibers.get(plugin.name)
        if existing is not None and existing.state is not FiberState.DISPOSED:
            raise PluginNameConflictError(plugin.name, existing.state)
        fiber = Fiber(plugin, self, config)
        self._fibers[plugin.name] = fiber
        # 计算初始 epoch 并尝试激活；装配期多次 provide 合并为一次下游传播。
        with self.batch():
            fiber._refresh()
        return fiber

    def notify(self, names: Iterable[str]) -> None:
        """响应式通知：事务内登记待处理键，事务外同步刷新。

        事务边界的意义见 :meth:`batch`——多键提供方卸载/装载时，逐键通知会让
        注入多键的依赖方在中间视图上反复装/卸；登记后统一排空，依赖方只按
        最终视图判定一次。
        """
        if self._batch_depth:
            self._deferred.update(names)
            return
        self._refresh_dependents(frozenset(names))

    def _refresh_dependents(self, names: frozenset[str]) -> None:
        """按插入顺序刷新注入命中 ``names`` 的 Fiber（已销毁者跳过）。"""
        if not names:
            return
        # 取元组快照：依赖方的 apply 可能经 Context.plugin 追加新 Fiber，
        # 直接迭代字典视图会触发「迭代期间尺寸变化」。
        for fiber in tuple(self._fibers.values()):
            if fiber.state == FiberState.DISPOSED:
                continue
            inject_names = fiber.inject_names
            if not inject_names:
                continue  # 无依赖的插件不受通知影响
            if inject_names & names:
                fiber._refresh()

    @contextlib.contextmanager
    def batch(self, *, flush: bool = True) -> Iterator[None]:
        """把一组变更合并为一次下游传播（可重入）。

        最外层退出时统一排空待处理通知；``flush=False`` 用于 ``dispose``——终态
        拆除不需要传播，待处理通知随拆除一起丢弃。即使事务体抛异常也排空：
        ``store`` 已被修改时，下游不能停留在过期视图（否则会出现「依赖全可解
        却 INACTIVE」的静默缺口）。
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                if flush:
                    self._flush_notifications()
                else:
                    self._deferred.clear()

    def _flush_notifications(self) -> None:
        """排空待处理通知，直到没有新的依赖变更产生。

        排空期间回填事务态：依赖方装配时产生的新供应继续入队，由下一轮统一
        处理，而不是递归刷新——这是「一次变更一次传播」的关键。轮次上限把病态
        环转成可诊断错误。
        """
        self._batch_depth += 1
        try:
            rounds = 0
            while self._deferred:
                rounds += 1
                if rounds > _MAX_FLUSH_ROUNDS:
                    pending = ", ".join(sorted(self._deferred))
                    raise RuntimeError(f"通知排空 {rounds} 轮仍未收敛，待处理键：{pending}")
                names = frozenset(self._deferred)
                self._deferred.clear()
                self._refresh_dependents(names)
        finally:
            self._batch_depth -= 1

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
        # 终态拆除：逆序卸载本身会撤销供应并产生通知，但此时传播已无意义，
        # 以不排空事务丢弃，避免刷新正在被销毁的 Fiber。
        with self.batch(flush=False):
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
