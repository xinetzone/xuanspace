"""插件生命周期状态机（Fiber）测试。"""

from okf.context import Context
from okf.plugin import _INACTIVE, Fiber, FiberState, InjectSpec, Plugin


def _make_plugin(name: str = "p", inject=None, provide=None, apply=None):
    def default_apply(ctx, config) -> None:
        return None

    return Plugin(
        name=name,
        apply=apply or default_apply,
        inject=inject or [],
        provide=provide or [],
    )


def test_fiber_state_enum_values():
    """FiberState 六态枚举值正确。"""
    assert FiberState.PENDING == 0
    assert FiberState.LOADING == 1
    assert FiberState.ACTIVE == 2
    assert FiberState.UNLOADING == 3
    assert FiberState.DISPOSED == 4
    assert FiberState.FAILED == 5


def test_inject_spec_and_plugin_fields():
    """InjectSpec / Plugin dataclass 字段正确。"""
    spec = InjectSpec("svc", config={"x": 1})
    assert spec.name == "svc"
    assert spec.config == {"x": 1}

    plugin = Plugin("p", lambda ctx, cfg: None, inject=[spec], provide=["a"])
    assert plugin.name == "p"
    assert plugin.inject == [spec]
    assert plugin.provide == ["a"]

    # 默认字段为空列表
    bare = Plugin("bare", lambda ctx, cfg: None)
    assert bare.inject == []
    assert bare.provide == []


def test_fiber_no_dependency_reload_becomes_active():
    """无依赖插件 _reload 后进入 ACTIVE。"""
    ctx = Context()
    fiber = Fiber(_make_plugin(), ctx)
    assert fiber.state == FiberState.PENDING
    fiber._reload()
    assert fiber.state == FiberState.ACTIVE
    assert fiber._error is None


def test_fiber_apply_raises_sets_failed():
    """apply 抛异常时 Fiber 进入 FAILED 并记录错误。"""
    ctx = Context()

    def bad_apply(ctx, config) -> None:
        raise RuntimeError("boom")

    fiber = Fiber(_make_plugin(apply=bad_apply), ctx)
    fiber._reload()
    assert fiber.state == FiberState.FAILED
    assert isinstance(fiber._error, RuntimeError)
    assert str(fiber._error) == "boom"


def test_epoch_valid_when_dependencies_satisfied():
    """依赖满足时 epoch 为有效实现标识拼接。"""
    ctx = Context()
    impl = object()
    ctx.provide("dep", impl)

    plugin = _make_plugin(inject=[InjectSpec("dep")])
    fiber = Fiber(plugin, ctx)
    assert fiber._compute_epoch() == f"dep:{id(impl)}"


def test_dependency_missing_fiber_stays_pending():
    """依赖缺失时 epoch 为 INACTIVE 哨兵，fiber 保持 PENDING。"""
    ctx = Context()
    plugin = _make_plugin(inject=[InjectSpec("dep")])
    fiber = Fiber(plugin, ctx)

    assert fiber._compute_epoch() == _INACTIVE
    fiber._refresh()
    assert fiber._epoch == _INACTIVE
    assert fiber.state == FiberState.PENDING


def test_fiber_unload_reverse_order():
    """Fiber._unload 逆序执行 disposable。"""
    ctx = Context()
    fiber = Fiber(_make_plugin(), ctx)
    order: list[str] = []

    fiber.effect(lambda: [lambda: order.append("a")])
    fiber._reload()
    assert fiber.state == FiberState.ACTIVE

    fiber._unload()
    # 只有一个效应，但验证清空与状态回退
    assert order == ["a"]
    assert fiber.state == FiberState.PENDING


def test_fiber_unload_multiple_effects_reverse_order():
    """多个效应逆序执行（后注册先回调）。"""
    ctx = Context()
    fiber = Fiber(_make_plugin(), ctx)
    order: list[str] = []

    fiber.effect(lambda: [lambda: order.append("first"), lambda: order.append("second")])
    fiber._reload()
    fiber._unload()
    assert order == ["second", "first"]
