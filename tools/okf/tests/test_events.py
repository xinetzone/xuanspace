"""事件系统（五种分发模式）测试。"""

import asyncio

from okf import events  # noqa: F401  # 导入即 monkey-patch Context 添加事件方法
from okf.context import Context


def test_emit_pure_notification_sync():
    """ctx.emit：纯通知，所有监听器同步执行，无返回值。"""
    ctx = Context()
    calls: list[tuple] = []

    ctx.on("e", lambda *a: calls.append(("h1", a)))
    ctx.on("e", lambda *a: calls.append(("h2", a)))

    result = ctx.emit("e", 1, 2)
    assert result is None
    assert calls == [("h1", (1, 2)), ("h2", (1, 2))]


def test_bail_short_circuits_on_first_non_none():
    """ctx.bail：第一个返回非 None 的结果短路。"""
    ctx = Context()
    called: list[str] = []

    def first():
        called.append("first")
        return None

    def second():
        called.append("second")
        return "result"

    def third():
        called.append("third")
        return "later"

    ctx.on("e", first)
    ctx.on("e", second)
    ctx.on("e", third)

    assert ctx.bail("e") == "result"
    assert called == ["first", "second"]


def test_serial_returns_first_non_none():
    """ctx.serial：串行执行，第一个非 None 的结果获胜。"""
    ctx = Context()
    order: list[str] = []

    def a():
        order.append("a")
        return None

    def b():
        order.append("b")
        return "win"

    def c():
        order.append("c")
        return "later"

    ctx.on("e", a)
    ctx.on("e", b)
    ctx.on("e", c)

    assert ctx.serial("e") == "win"
    assert order == ["a", "b"]


def test_waterfall_middleware_chain():
    """ctx.waterfall：中间件链，next() 继续传递。"""
    ctx = Context()
    order: list[tuple] = []

    def m1(value, next):
        order.append(("m1", value))
        return next(value + 1)

    def m2(value, next):
        order.append(("m2", value))
        return value * 10

    ctx.on("e", m1)
    ctx.on("e", m2)

    assert ctx.waterfall("e", 0) == 10
    assert order == [("m1", 0), ("m2", 1)]


def test_waterfall_short_circuit_without_next():
    """不调用 next() 时短路，后续中间件不再执行。"""
    ctx = Context()

    def stop(value, next):
        return "stopped"

    def never(value, next):
        raise AssertionError("should not be called")

    ctx.on("e", stop)
    ctx.on("e", never)

    assert ctx.waterfall("e", "x") == "stopped"


def test_parallel_runs_async_listeners():
    """ctx.parallel：异步监听器并行执行。"""
    ctx = Context()

    async def h1():
        await asyncio.sleep(0.01)
        return 1

    async def h2():
        await asyncio.sleep(0.01)
        return 2

    ctx.on("e", h1)
    ctx.on("e", h2)

    assert ctx.parallel("e") == [1, 2]


def test_parallel_mixed_sync_and_async():
    """ctx.parallel 兼容同步与异步监听器混合。"""
    ctx = Context()

    async def h_async():
        await asyncio.sleep(0.01)
        return "async"

    def h_sync():
        return "sync"

    ctx.on("e", h_async)
    ctx.on("e", h_sync)

    assert ctx.parallel("e") == ["async", "sync"]


def test_on_returns_disposable_to_cancel():
    """ctx.on 返回 Disposable，可取消监听。"""
    ctx = Context()
    calls: list[tuple] = []

    def handler(*a):
        calls.append(a)

    undo = ctx.on("e", handler)

    ctx.emit("e")
    assert calls == [()]

    undo()
    ctx.emit("e")
    assert calls == [()]  # 取消后不再触发
