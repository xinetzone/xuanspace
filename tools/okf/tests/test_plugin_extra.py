"""插件生命周期状态机（Fiber）补充测试：epoch 切换、dispose、效应逆函数。"""

from okf.context import Context
from okf.plugin import _INACTIVE, Fiber, FiberState, InjectSpec, Plugin


def _mk_plugin(name="p", inject=None, provide=None, apply=None):
    def default_apply(ctx, config):
        return None

    return Plugin(
        name=name,
        apply=apply or default_apply,
        inject=inject or [],
        provide=provide or [],
    )


def test_fiber_name_property() -> None:
    ctx = Context()
    fiber = Fiber(_mk_plugin(name="consumer"), ctx)
    assert fiber.name == "consumer"


def test_effect_single_callable() -> None:
    """execute 返回单个逆函数（非 list）时被自动包装为列表。"""
    ctx = Context()
    fiber = Fiber(_mk_plugin(), ctx)
    order: list[str] = []
    undo = fiber.effect(lambda: lambda: order.append("only"))
    assert len(fiber._disposables) == 1
    undo()
    assert order == ["only"]


def test_effect_undo_swallows_exception() -> None:
    """逆函数执行时抛异常被吞掉，不回传。"""
    ctx = Context()
    fiber = Fiber(_mk_plugin(), ctx)

    def boom():
        raise RuntimeError("boom")

    undo = fiber.effect(lambda: [boom])
    undo()  # 不应抛出


def test_epoch_change_unload_then_reload() -> None:
    """依赖实现切换触发 _unload 后 _reload（epoch 变更）。"""
    ctx = Context()
    impl1 = object()
    ctx.provide("dep", impl1)

    loaded: list[object] = []

    def apply(ctx, config):
        loaded.append(ctx.get("dep"))

    fiber = Fiber(_mk_plugin(inject=[InjectSpec("dep")], apply=apply), ctx)
    fiber._refresh()
    assert fiber.state == FiberState.ACTIVE
    assert loaded == [impl1]

    impl2 = object()
    ctx.provide("dep", impl2)  # 替换实现
    fiber._refresh()
    assert fiber.state == FiberState.ACTIVE
    assert loaded == [impl1, impl2]
    assert fiber._epoch == f"dep:{id(impl2)}"


def test_unload_swallows_dispose_exception() -> None:
    """卸载时单个 disposable 抛异常被吞掉，其余仍执行，状态回退 PENDING。"""
    ctx = Context()
    fiber = Fiber(_mk_plugin(), ctx)
    order: list[str] = []

    def boom():
        raise RuntimeError("boom")

    fiber.effect(lambda: [boom, lambda: order.append("ok")])
    fiber._reload()
    fiber._unload()  # 不应抛出
    assert order == ["ok"]
    assert fiber.state == FiberState.PENDING


def test_dispose_active() -> None:
    ctx = Context()
    fiber = Fiber(_mk_plugin(), ctx)
    fiber._reload()
    assert fiber.state == FiberState.ACTIVE
    fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


def test_dispose_failed() -> None:
    ctx = Context()

    def bad_apply(ctx, config):
        raise RuntimeError("boom")

    fiber = Fiber(_mk_plugin(apply=bad_apply), ctx)
    fiber._reload()
    assert fiber.state == FiberState.FAILED
    fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


def test_dispose_pending() -> None:
    """PENDING 状态 dispose 不经过 _unload，直接 DISPOSED。"""
    ctx = Context()
    fiber = Fiber(_mk_plugin(), ctx)
    assert fiber.state == FiberState.PENDING
    fiber.dispose()
    assert fiber.state == FiberState.DISPOSED


def test_failed_recovery_and_epoch_change() -> None:
    """FAILED 状态恢复：注入依赖后 epoch 从 INACTIVE → 有效，_reload 重新激活。"""
    ctx = Context()
    attempts: list[int] = []

    def apply(ctx, config):
        attempts.append(ctx.get("dep"))
        if len(attempts) < 2:
            raise RuntimeError("transient")

    fiber = Fiber(_mk_plugin(inject=[InjectSpec("dep")], apply=apply), ctx)
    # 依赖缺失 → INACTIVE
    fiber._refresh()
    assert fiber._epoch == _INACTIVE
    assert fiber.state == FiberState.PENDING

    # 提供依赖 → 激活成功（首次 apply 失败）
    impl = object()
    ctx.provide("dep", impl)
    fiber._refresh()
    assert fiber.state == FiberState.FAILED
    assert isinstance(fiber._error, RuntimeError)

    # 重新提供新的实现变更 epoch，_refresh 后重新 apply 成功
    impl2 = object()
    ctx.provide("dep", impl2)
    fiber._refresh()
    assert fiber.state == FiberState.ACTIVE
