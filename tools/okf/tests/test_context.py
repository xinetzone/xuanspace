"""统一上下文（Context）测试。"""

import pytest

from okf.context import Context
from okf.plugin import FiberState, InjectSpec, Plugin


def test_provide_get_and_undo():
    """provide 后能获取，撤销后抛 KeyError。"""
    ctx = Context()
    service = object()
    undo = ctx.provide("svc", service)
    assert ctx.get("svc") is service

    undo()
    with pytest.raises(KeyError):
        ctx.get("svc")


def test_provide_notify_activates_dependent_plugin():
    """provide 服务后，依赖该服务的插件自动激活。"""
    ctx = Context()
    received: list[object] = []

    def apply(ctx, config) -> None:
        received.append(ctx.get("dep"))

    plugin = Plugin("consumer", apply, inject=[InjectSpec("dep")])
    ctx.plugin(plugin)
    # 初始依赖缺失，保持 PENDING
    assert ctx._fibers["consumer"].state == FiberState.PENDING

    impl = object()
    ctx.provide("dep", impl)
    assert ctx._fibers["consumer"].state == FiberState.ACTIVE
    assert received == [impl]


def test_provide_undo_deactivates_dependent_plugin():
    """撤销供应后，依赖插件被停用。"""
    ctx = Context()

    def apply(ctx, config) -> None:
        ctx.effect(lambda: [lambda: None])

    plugin = Plugin("consumer", apply, inject=[InjectSpec("dep")])
    ctx.plugin(plugin)

    impl = object()
    undo = ctx.provide("dep", impl)
    assert ctx._fibers["consumer"].state == FiberState.ACTIVE

    undo()
    assert ctx._fibers["consumer"].state == FiberState.PENDING


def test_bundle_loader_activates_conformance_checker(sample_bundle_path):
    """插件装配端到端：bundle_loader 提供 bundle_accessor → conformance_checker 自动激活。"""
    from okf.plugins.conformance import ConformanceChecker
    from okf.plugins.loader import BundleLoader

    ctx = Context()
    ctx.plugin(ConformanceChecker())
    ctx.plugin(BundleLoader(), {"path": str(sample_bundle_path)})

    assert ctx.get("bundle_accessor") is not None
    assert ctx.get("conformance_report") is not None
    assert ctx._fibers["conformance_checker"].state == FiberState.ACTIVE


def test_context_effect_registers_undo():
    """Context.effect 注册可逆效应，返回逆函数。"""
    ctx = Context()
    calls: list[str] = []

    undo = ctx.effect(lambda: lambda: calls.append("effect"), label="test")
    assert len(ctx._root_disposables) == 1
    assert ctx.get_effects()[0].label == "test"

    undo()
    assert calls == ["effect"]
