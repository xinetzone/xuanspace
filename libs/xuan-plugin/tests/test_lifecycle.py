"""生命周期状态机（P2-4）测试：INACTIVE 原因、显式重试、终态与静态生命周期。

验收依据设计文档 §5.5 的状态表与不变量：
PENDING/INACTIVE 无效应、INACTIVE 原因可解释、FAILED 可显式重试、
DISPOSED 为终态且二次 dispose 幂等；静态插件 ACTIVE 后不热换（ADR-03）。
"""

import pytest
from xuan_plugin.context import Context
from xuan_plugin.plugin import FiberState, FiberTransitionError, InjectSpec, Plugin


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

    ctx.provide("dep", object())
    assert fiber.state is FiberState.ACTIVE
    assert calls == ["native"]

    # 依赖实现变更：静态插件拒绝先卸后载，保持 ACTIVE 且不再 apply
    ctx.provide("dep", object())
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
    ctx.provide("dep", object())

    def apply(ctx, config) -> None:
        ctx.provide("dep", object())  # 自依赖写入会触发对本 fiber 的 notify

    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert fiber.state is FiberState.ACTIVE
