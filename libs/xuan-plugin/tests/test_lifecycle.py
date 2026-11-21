"""生命周期状态机（P2-4）测试：INACTIVE 原因、显式重试、终态与静态生命周期。

验收依据设计文档 §5.5 的状态表与不变量：
PENDING/INACTIVE 无效应、INACTIVE 原因可解释、FAILED 可显式重试、
DISPOSED 为终态且二次 dispose 幂等；静态插件 ACTIVE 后不热换（ADR-03）。
"""

import pytest
from xuan_plugin.context import Context
from xuan_plugin.plugin import _INACTIVE, _TRANSITIONS, Fiber, FiberState, FiberTransitionError, InjectSpec, Plugin

# 设计 §5.5 迁移表；另含代码注释显式补入的三条运行期边：
# FAILED → INACTIVE（依赖消失）、FAILED → UNLOADING（dispose 回收失败节点）、
# UNLOADING → FAILED（撤销异常）。其余 7x7 组合一律非法。
_ALLOWED_TRANSITIONS = frozenset(
    {
        (FiberState.PENDING, FiberState.LOADING),
        (FiberState.PENDING, FiberState.INACTIVE),
        (FiberState.PENDING, FiberState.DISPOSED),
        (FiberState.INACTIVE, FiberState.LOADING),
        (FiberState.INACTIVE, FiberState.DISPOSED),
        (FiberState.LOADING, FiberState.ACTIVE),
        (FiberState.LOADING, FiberState.FAILED),
        (FiberState.ACTIVE, FiberState.UNLOADING),
        (FiberState.ACTIVE, FiberState.DISPOSED),
        (FiberState.UNLOADING, FiberState.PENDING),
        (FiberState.UNLOADING, FiberState.FAILED),
        (FiberState.FAILED, FiberState.LOADING),
        (FiberState.FAILED, FiberState.INACTIVE),
        (FiberState.FAILED, FiberState.UNLOADING),
        (FiberState.FAILED, FiberState.DISPOSED),
    }
)


class DepV1:
    """依赖实现 V1：类身份参与摘要。"""


class DepV2:
    """依赖实现 V2：与 V1 摘要不同，用于触发 epoch 变更。"""


def _plugin(
    name: str = "p",
    *,
    inject: tuple[str, ...] = (),
    lifecycle: str = "reloadable",
    calls: list[str] | None = None,
) -> Plugin:
    def apply(ctx, config) -> None:
        if calls is not None:
            calls.append(name)
        for key in inject:
            ctx.get(key)  # 未满足依赖时本行不可达，保留以证明 apply 只在可解时执行

    return Plugin(
        name=name,
        apply=apply,
        inject=[InjectSpec(key) for key in inject],
        lifecycle=lifecycle,
    )


def test_state_values_stay_backward_compatible() -> None:
    """INACTIVE 追加在末尾，既有六态数值不漂移。"""
    assert FiberState.PENDING == 0
    assert FiberState.FAILED == 5
    assert FiberState.INACTIVE == 6


def test_missing_dependency_enters_inactive_with_reason() -> None:
    ctx = Context()
    fiber = ctx.plugin(_plugin(inject=("dep",)))

    assert fiber.state is FiberState.INACTIVE
    assert fiber.reason == ("dep",)
    assert len(fiber._disposables) == 0  # INACTIVE 不持有效应


def test_reason_updates_when_partial_dependency_appears() -> None:
    ctx = Context()
    fiber = ctx.plugin(_plugin(inject=("a", "b")))
    assert fiber.reason == ("a", "b")

    ctx.provide("a", object())
    assert fiber.state is FiberState.INACTIVE
    assert fiber.reason == ("b",)


def test_dependency_appearance_and_loss_transition() -> None:
    ctx = Context()
    fiber = ctx.plugin(_plugin(inject=("dep",)))
    assert fiber.state is FiberState.INACTIVE

    undo = ctx.provide("dep", object())
    assert fiber.state is FiberState.ACTIVE
    assert fiber.reason == ()

    undo()
    assert fiber.state is FiberState.INACTIVE
    assert fiber.reason == ("dep",)


def test_failed_plugin_can_retry_explicitly() -> None:
    state = {"fail": True}

    def apply(ctx, config) -> None:
        if state["fail"]:
            raise RuntimeError("transient")

    ctx = Context()
    ctx.provide("dep", object())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert fiber.state is FiberState.FAILED
    assert isinstance(fiber.error, RuntimeError)

    state["fail"] = False
    fiber.retry()

    assert fiber.state is FiberState.ACTIVE
    assert fiber.error is None


def test_retry_is_only_legal_from_failed() -> None:
    ctx = Context()
    active = ctx.plugin(_plugin("active"))

    with pytest.raises(FiberTransitionError, match="ACTIVE"):
        active.retry()


def test_disposed_is_terminal_and_idempotent() -> None:
    ctx = Context()
    fiber = ctx.plugin(_plugin("done"))
    fiber.dispose()
    assert fiber.state is FiberState.DISPOSED

    fiber.dispose()  # 幂等
    assert fiber.state is FiberState.DISPOSED

    with pytest.raises(FiberTransitionError, match="DISPOSED"):
        fiber._refresh()
    with pytest.raises(FiberTransitionError, match="DISPOSED"):
        fiber.retry()


def test_static_plugin_suppresses_hot_reload() -> None:
    calls: list[str] = []
    ctx = Context()
    fiber = ctx.plugin(_plugin("native", inject=("dep",), lifecycle="static", calls=calls))
    assert fiber.state is FiberState.INACTIVE

    ctx.provide("dep", DepV1())
    assert fiber.state is FiberState.ACTIVE
    assert calls == ["native"]

    # 依赖实现变更：静态插件拒绝先卸后载，保持 ACTIVE 且不再 apply
    ctx.provide("dep", DepV2())
    assert fiber.state is FiberState.ACTIVE
    assert calls == ["native"]
    assert fiber.suppressed_reloads == 1


def test_static_plugin_still_retires_handles_on_dispose() -> None:
    undone: list[str] = []

    def apply(ctx, config) -> None:
        ctx.effect(lambda: [lambda: undone.append("undo")])

    ctx = Context()
    fiber = ctx.plugin(Plugin("native", apply, lifecycle="static"))
    assert fiber.state is FiberState.ACTIVE

    assert ctx.dispose() == []
    assert fiber.state is FiberState.DISPOSED
    assert undone == ["undo"]


def test_unknown_lifecycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="生命周期"):
        Plugin("p", lambda ctx, cfg: None, lifecycle="hot")


def test_refresh_during_apply_does_not_reenter_state_machine() -> None:
    """apply 内部 provide 触发的 notify 不得把 LOADING 再迁一次。"""
    ctx = Context()
    ctx.provide("dep", DepV1())

    def apply(ctx, config) -> None:
        ctx.provide("dep", DepV2())  # 自依赖写入会触发对本 fiber 的 notify

    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert fiber.state is FiberState.ACTIVE


def test_transition_table_matches_documented_edges() -> None:
    """迁移表与设计 §5.5 逐项一致，且终态无出边。"""
    assert set(_TRANSITIONS) == set(FiberState)
    assert _TRANSITIONS[FiberState.DISPOSED] == frozenset()
    actual = {(source, target) for source, targets in _TRANSITIONS.items() for target in targets}
    assert actual == set(_ALLOWED_TRANSITIONS)


def test_transition_rejects_every_edge_outside_the_table() -> None:
    """穷举 7x7：表内迁移成功，表外必抛 FiberTransitionError 且状态不变。"""
    for source in FiberState:
        for target in FiberState:
            fiber = Fiber(Plugin(name="p", apply=lambda ctx, cfg: None), Context())
            fiber.state = source
            if (source, target) in _ALLOWED_TRANSITIONS:
                fiber._transition(target)
                assert fiber.state is target
            else:
                with pytest.raises(FiberTransitionError):
                    fiber._transition(target)
                assert fiber.state is source


def test_non_active_states_hold_no_effects() -> None:
    """§5.5 不变量：PENDING / INACTIVE / FAILED / DISPOSED 下效应数恒为 0。"""
    pending = Fiber(Plugin(name="pending", apply=lambda ctx, cfg: None), Context())
    assert pending.state is FiberState.PENDING
    assert len(pending._disposables) == 0

    inactive = Fiber(
        Plugin(name="inactive", apply=lambda ctx, cfg: None, inject=[InjectSpec("missing")]),
        Context(),
    )
    inactive._refresh()
    assert inactive.state is FiberState.INACTIVE
    assert len(inactive._disposables) == 0

    def boom(ctx: Context, config: dict) -> None:
        ctx.effect(lambda: None)
        raise RuntimeError("boom")

    failed = Fiber(Plugin(name="failed", apply=boom), Context())
    failed._refresh()
    assert failed.state is FiberState.FAILED
    assert len(failed._disposables) == 0

    disposed = Fiber(Plugin(name="disposed", apply=lambda ctx, cfg: None), Context())
    disposed.dispose()
    assert disposed.state is FiberState.DISPOSED
    assert len(disposed._disposables) == 0


def test_active_implies_resolvable_inputs() -> None:
    """§5.5 不变量：ACTIVE ⇒ 输入快照可解（``_inputs is not None`` 且 epoch 非 ``_INACTIVE``）。"""

    def assert_invariant(fiber: Fiber) -> None:
        if fiber.state is FiberState.ACTIVE:
            assert fiber._inputs is not None
            assert fiber._epoch != _INACTIVE

    solo = Fiber(Plugin(name="solo", apply=lambda ctx, cfg: None), Context())
    solo._refresh()
    assert solo.state is FiberState.ACTIVE
    assert_invariant(solo)

    ctx = Context()
    ctx.provide("dep", DepV1())
    with_dep = Fiber(Plugin(name="p", apply=lambda ctx, cfg: None, inject=[InjectSpec("dep")]), ctx)
    with_dep._refresh()
    assert with_dep.state is FiberState.ACTIVE
    assert_invariant(with_dep)
