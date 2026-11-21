"""下游传播批处理（P2-6）测试：事务边界、合并排空、级联收敛与终态拆除。

验收依据设计文档 §5.4 第 5–9 步与度量 M4：
- 一次逻辑变更（批量 provide 或单个提供方热替换）只产生一轮下游传播；
- 依赖方只在最终视图上判定，不会看到「键暂时缺失」的中间态；
- 依赖恢复且身份未变时，``INACTIVE`` 必须收敛回 ``ACTIVE``；
- 事务外 ``provide`` 保持同步语义不变。
"""

import pytest
from xuan_plugin.context import Context
from xuan_plugin.plugin import FiberState, InjectSpec, Plugin


class ImplV1:
    """实现 V1（类身份参与摘要，避免地址复用造成的伪相等）。"""


class ImplV2:
    """实现 V2：与 V1 摘要不同。"""


class DepA1:
    pass


class DepA2:
    pass


def _provider(name: str, key: str, dep: str, holder: dict) -> Plugin:
    """提供 ``key`` 的插件；``holder["impl"]`` 决定提供哪一版实现。"""

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide(key, holder["impl"]()))

    return Plugin(name, apply, inject=[InjectSpec(dep)])


def _consumer(
    name: str,
    inject: list[str],
    provide: tuple[tuple[str, dict], ...] = (),
    log: list[str] | None = None,
) -> Plugin:
    """注入 ``inject``、可选供应 ``provide`` 的依赖方，装/卸都记入 ``log``。"""
    events = log if log is not None else []

    def apply(ctx, config) -> None:
        events.append(f"{name}:apply")
        for key, holder in provide:
            ctx.effect(lambda k=key, h=holder: ctx.provide(k, h["impl"]()))
        ctx.effect(lambda: [lambda: events.append(f"{name}:undo")])

    return Plugin(name, apply, inject=[InjectSpec(k) for k in inject])


def test_batch_defers_propagation_until_outermost_exit() -> None:
    """事务内只登记，最外层退出才刷新。"""
    ctx = Context()
    fiber = ctx.plugin(_consumer("C", ["dep"]))
    assert fiber.state is FiberState.INACTIVE

    with ctx.batch():
        ctx.provide("dep", ImplV1())
        assert fiber.state is FiberState.INACTIVE

    assert fiber.state is FiberState.ACTIVE


def test_batch_is_reentrant_and_flushes_once() -> None:
    """嵌套事务只在最外层排空一次。"""
    log: list[str] = []
    ctx = Context()
    fiber = ctx.plugin(_consumer("C", ["dep"], log=log))

    with ctx.batch():
        ctx.provide("dep", ImplV1())
        with ctx.batch():
            ctx.provide("unrelated", ImplV2())
        assert fiber.state is FiberState.INACTIVE

    assert fiber.state is FiberState.ACTIVE
    assert log == ["C:apply"]


def test_batch_flushes_even_if_body_raises() -> None:
    """事务体抛异常时仍排空：store 已变更，下游不能停在过期视图。"""
    ctx = Context()
    fiber = ctx.plugin(_consumer("C", ["dep"]))

    with pytest.raises(RuntimeError):
        with ctx.batch():
            ctx.provide("dep", ImplV1())
            raise RuntimeError("body failed")

    assert fiber.state is FiberState.ACTIVE


def test_multi_key_provider_reload_is_single_cycle() -> None:
    """一个提供方提供 2 键：热替换后依赖方只卸一次、装一次。"""
    log: list[str] = []
    holder = {"impl": ImplV1}
    ctx = Context()
    ctx.provide("dep", DepA1())

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide("k1", holder["impl"]()))
        ctx.effect(lambda: ctx.provide("k2", holder["impl"]()))

    ctx.plugin(Plugin("P", apply, inject=[InjectSpec("dep")]))
    fiber_c = ctx.plugin(_consumer("C", ["k1", "k2"], log=log))
    assert fiber_c.state is FiberState.ACTIVE
    log.clear()

    holder["impl"] = ImplV2
    ctx.provide("dep", DepA2())

    assert log == ["C:undo", "C:apply"]
    assert fiber_c.state is FiberState.ACTIVE


def test_two_providers_in_one_batch_single_cycle() -> None:
    """两位提供方在同一批内变更：依赖方只经历一次替换（M4）。"""
    log: list[str] = []
    holder_a = {"impl": ImplV1}
    holder_b = {"impl": ImplV1}
    ctx = Context()
    ctx.provide("dep_a", DepA1())
    ctx.provide("dep_b", DepA1())
    ctx.plugin(_provider("A", "k1", "dep_a", holder_a))
    ctx.plugin(_provider("B", "k2", "dep_b", holder_b))
    fiber_c = ctx.plugin(_consumer("C", ["k1", "k2"], log=log))
    assert fiber_c.state is FiberState.ACTIVE
    log.clear()

    holder_a["impl"] = ImplV2
    holder_b["impl"] = ImplV2
    with ctx.batch():
        ctx.provide("dep_a", DepA2())
        ctx.provide("dep_b", DepA2())
        assert fiber_c.state is FiberState.ACTIVE  # 批内无中间态

    assert log == ["C:undo", "C:apply"]
    assert fiber_c.reloads == 1


def test_resolvable_inactive_fiber_reactivates() -> None:
    """撤销后重供同一实现：不得停在「依赖全可解却 INACTIVE」。"""
    log: list[str] = []
    ctx = Context()
    impl = ImplV1()
    undo = ctx.provide("dep", impl)
    fiber = ctx.plugin(_consumer("C", ["dep"], log=log))
    assert fiber.state is FiberState.ACTIVE
    log.clear()

    undo()
    assert fiber.state is FiberState.INACTIVE

    ctx.provide("dep", impl)  # 同一实现 → 摘要不变
    assert fiber.state is FiberState.ACTIVE
    assert log == ["C:undo", "C:apply"]


def test_cascade_flushes_new_notifications_in_next_round() -> None:
    """级联：排空期间依赖方新供应的键由下一轮处理。"""
    log: list[str] = []
    p_holder = {"impl": ImplV1}
    holder = {"impl": ImplV1}
    ctx = Context()
    ctx.provide("dep", DepA1())
    ctx.plugin(_provider("P", "k1", "dep", p_holder))
    ctx.plugin(_consumer("C", ["k1"], provide=(("m1", holder),), log=log))
    fiber_d = ctx.plugin(_consumer("D", ["m1"], log=log))
    assert fiber_d.state is FiberState.ACTIVE
    log.clear()

    p_holder["impl"] = ImplV2
    holder["impl"] = ImplV2  # C 重装配时供应新实现 → 第二轮触发 D
    ctx.provide("dep", DepA2())

    assert log == ["C:undo", "C:apply", "D:undo", "D:apply"]
    assert fiber_d.state is FiberState.ACTIVE


def test_static_plugin_suppression_counts_once_per_batch() -> None:
    """静态插件的抑制计数按逻辑变更计一次，而非按每个键各计一次。"""
    holder_a = {"impl": ImplV1}
    holder_b = {"impl": ImplV1}
    ctx = Context()
    ctx.provide("dep_a", DepA1())
    ctx.provide("dep_b", DepA1())
    ctx.plugin(_provider("A", "k1", "dep_a", holder_a))
    ctx.plugin(_provider("B", "k2", "dep_b", holder_b))
    static = Plugin(
        "S",
        lambda ctx, config: None,
        inject=[InjectSpec("k1"), InjectSpec("k2")],
        lifecycle="static",
    )
    fiber_s = ctx.plugin(static)
    assert fiber_s.state is FiberState.ACTIVE

    holder_a["impl"] = ImplV2
    holder_b["impl"] = ImplV2
    with ctx.batch():
        ctx.provide("dep_a", DepA2())
        ctx.provide("dep_b", DepA2())

    assert fiber_s.state is FiberState.ACTIVE
    assert fiber_s.suppressed_reloads == 1


def test_dispose_discards_pending_propagation() -> None:
    """dispose 是终态拆除：卸载产生的通知不再传播。"""
    ctx = Context()
    ctx.provide("dep", ImplV1())
    ctx.plugin(_provider("P", "k1", "dep", {"impl": ImplV1}))
    fiber = ctx.plugin(_consumer("C", ["k1"]))
    assert fiber.state is FiberState.ACTIVE

    errors = ctx.dispose()

    assert errors == []
    assert fiber.state is FiberState.DISPOSED
    assert ctx._deferred == set()
