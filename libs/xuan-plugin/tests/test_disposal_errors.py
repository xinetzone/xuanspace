"""卸载异常聚合（P0-3）测试。

验收依据设计文档 §7 P0-3：注入故障插件后断言「effect 数 = 0 且异常被上报」。
"""

from xuan_plugin.context import Context
from xuan_plugin.disposable import DisposableList, DisposalError
from xuan_plugin.plugin import Fiber, FiberState, Plugin


def _boom() -> None:
    raise RuntimeError("boom")


def _ok() -> None:
    pass


def _make_fiber(ctx: Context, name: str = "p") -> Fiber:
    return Fiber(Plugin(name=name, apply=lambda c, cfg: None), ctx)


def test_dispose_all_continues_after_failure() -> None:
    """首个撤销失败不应中断同批后续撤销。"""
    calls: list[str] = []
    items = DisposableList()
    items.push(lambda: calls.append("first"))
    items.push(_boom)
    items.push(lambda: calls.append("last"))

    errors = items.dispose_all()

    assert len(errors) == 1
    assert calls == ["last", "first"]  # 逆序执行，且首个失败未阻断
    assert len(items) == 0


def test_dispose_all_reports_each_failure() -> None:
    """report 回调对每个失败项各触发一次。"""
    reported: list[str] = []
    items = DisposableList()
    items.push(_boom)
    items.push(_boom)

    errors = items.dispose_all(report=lambda label, exc: reported.append(label))

    assert len(errors) == 2
    assert reported == ["_boom", "_boom"]


def test_unload_aggregates_and_zeroes_effects() -> None:
    """卸载后效应数归零，且失败被聚合为 DisposalError。"""
    ctx = Context()
    fiber = _make_fiber(ctx)
    fiber._refresh()
    fiber.effect(lambda: _boom)
    fiber.effect(lambda: _ok)
    assert len(fiber._disposables) == 2

    fiber._unload()

    assert len(fiber._disposables) == 0
    assert fiber.state == FiberState.FAILED
    assert isinstance(fiber.error, DisposalError)
    assert len(fiber.error.errors) == 1


def test_reload_failure_rolls_back_effects() -> None:
    """apply 中途失败时已登记 effect 必须被强制撤销。"""

    def apply(c: Context, cfg: dict) -> None:
        c.effect(lambda: _ok)
        raise RuntimeError("boom")

    ctx = Context()
    fiber = Fiber(Plugin(name="p", apply=apply), ctx)
    fiber._refresh()

    assert fiber.state == FiberState.FAILED
    assert len(fiber._disposables) == 0
    assert isinstance(fiber.error, RuntimeError)
    assert str(fiber.error) == "boom"


def test_dispose_keeps_failed_state() -> None:
    """卸载失败的 Fiber 保留 FAILED，不静默降级为 DISPOSED。"""
    ctx = Context()
    fiber = _make_fiber(ctx)
    fiber._refresh()
    fiber.effect(lambda: _boom)

    fiber.dispose()

    assert fiber.state == FiberState.FAILED
    assert isinstance(fiber.error, DisposalError)


def test_context_dispose_collects_fiber_errors() -> None:
    """Context.dispose 汇总 Fiber 卸载失败，且不抛出。"""
    ctx = Context()
    ctx.plugin(Plugin(name="bad", apply=lambda c, cfg: None))
    ctx._fibers["bad"].effect(lambda: _boom)

    errors = ctx.dispose()

    assert len(errors) == 1
    assert errors[0][0] == "fiber:bad"
    assert ctx.get_disposal_errors() == errors


def test_context_dispose_collects_root_errors() -> None:
    """Context.dispose 汇总根效应失败，标签带 root 前缀。"""
    ctx = Context()
    ctx.effect(lambda: _boom)

    errors = ctx.dispose()

    assert len(errors) == 1
    assert errors[0][0] == "root:_boom"
    assert ctx.get_disposal_errors() == errors
    assert len(ctx._root_disposables) == 0
