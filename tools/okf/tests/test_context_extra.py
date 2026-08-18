"""统一上下文（Context）补充测试：notify 边界、dispose、根效应。"""

from okf.context import Context
from okf.plugin import FiberState, InjectSpec, Plugin


def test_effect_undo_swallows_exception() -> None:
    """根效应逆函数抛异常被吞掉。"""
    ctx = Context()

    def boom():
        raise RuntimeError("boom")

    undo = ctx.effect(lambda: [boom], label="bad")
    undo()  # 不应抛出


def test_effect_without_label_not_recorded() -> None:
    """无 label 的根效应不记录 EffectMeta。"""
    ctx = Context()
    ctx.effect(lambda: [lambda: None])
    assert ctx.get_effects() == []


def test_notify_skips_disposed_fiber() -> None:
    """已 dispose 的 fiber 不响应 notify。"""
    ctx = Context()

    def apply(ctx, config):
        raise AssertionError("disposed fiber must not be activated")

    plugin = Plugin("consumer", apply, inject=[InjectSpec("dep")])
    ctx.plugin(plugin)
    fiber = ctx._fibers["consumer"]
    assert fiber.state == FiberState.PENDING

    fiber.dispose()
    assert fiber.state == FiberState.DISPOSED

    ctx.provide("dep", object())
    # 仍为 DISPOSED，未被重新激活
    assert fiber.state == FiberState.DISPOSED


def test_notify_ignores_plugin_without_deps() -> None:
    """无依赖插件不受 notify 影响（保持 ACTIVE，不被误停）。"""
    ctx = Context()
    plugin = Plugin("standalone", lambda ctx, cfg: None)
    ctx.plugin(plugin)
    assert ctx._fibers["standalone"].state == FiberState.ACTIVE

    ctx.notify(["some_service"])
    assert ctx._fibers["standalone"].state == FiberState.ACTIVE


def test_dispose_clears_everything() -> None:
    """dispose 卸载所有 fiber、清空服务与效应、执行根效应逆函数。"""
    ctx = Context()
    order: list[str] = []

    ctx.provide("svc", object())
    ctx.effect(lambda: [lambda: order.append("root-effect")], label="root")

    def apply(ctx, config):
        ctx.effect(lambda: [lambda: order.append("fiber-effect")])

    ctx.plugin(Plugin("consumer", apply, inject=[]))
    assert ctx._fibers["consumer"].state == FiberState.ACTIVE

    ctx.dispose()

    assert ctx._fibers == {}
    assert ctx._store == {}
    assert ctx._effects == []
    assert len(ctx._root_disposables) == 0
    assert order == ["fiber-effect", "root-effect"]


def test_provide_undo_does_not_remove_replacement() -> None:
    """撤销供应时若当前实现已被替换，则不删除新实现。"""
    ctx = Context()
    impl1 = object()
    impl2 = object()
    undo = ctx.provide("svc", impl1)
    ctx.provide("svc", impl2)
    undo()
    assert ctx.get("svc") is impl2


def test_context_manager_enter_exit_disposes() -> None:
    """with 语句：__enter__ 返回自身，__exit__ 自动 dispose 并清空 store。"""
    ctx = Context()
    ctx.provide("svc", object())

    with ctx as entered:
        assert entered is ctx
        assert ctx.get("svc") is not None

    # __exit__ 已调用 dispose：服务清空
    assert ctx._store == {}
    assert ctx._fibers == {}
