"""热更新事务（P2-5）测试：效应归属、事务化替换、失败回滚与观测。

验收依据设计文档 §5.4 第 6–8 步与 §5.5 状态表：
- ``apply`` 期间登记的效应归属 Fiber，卸载/替换时必定撤销；
- 依赖 epoch 变化触发「先撤旧、再装新」，成功计入 ``reloads`` 并给出逐字段变化；
- 新实现失败时撤销其部分效应（零残留），以 FAILED + 原始异常暴露并可 ``retry()``；
- 依赖消失时效应归零并进入 INACTIVE。
"""

from xuan_plugin.context import Context
from xuan_plugin.plugin import FiberState, InjectSpec, Plugin


class ImplV1:
    """依赖实现 V1：类身份参与摘要。"""


class ImplV2:
    """依赖实现 V2：与 V1 摘要不同，用于触发热替换。"""


def test_apply_effects_belong_to_fiber() -> None:
    """apply 内的效应归属当前 Fiber，不再沉淀到根上下文。"""
    undone: list[str] = []

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide("own", 1))
        ctx.effect(lambda: [lambda: undone.append("undo")])

    ctx = Context()
    fiber = ctx.plugin(Plugin("p", apply))
    assert fiber.state is FiberState.ACTIVE
    assert len(fiber._disposables) == 2
    assert len(ctx._root_disposables) == 0
    assert ctx.get("own") == 1

    fiber._unload()

    assert "own" not in ctx.registered_keys()
    assert undone == ["undo"]
    assert len(fiber._disposables) == 0


def test_hot_reload_undoes_old_and_applies_new() -> None:
    """依赖实现变更：旧供应被撤销，新供应生效，并记录变化字段。"""
    current: dict[str, object] = {}

    def apply(ctx, config) -> None:
        impl = object()
        current["impl"] = impl
        ctx.effect(lambda: ctx.provide("own", impl))

    ctx = Context()
    ctx.provide("dep", ImplV1())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    first = current["impl"]
    assert ctx.get("own") is first

    ctx.provide("dep", ImplV2())  # notify → 热替换

    assert fiber.state is FiberState.ACTIVE
    assert ctx.get("own") is current["impl"]
    assert ctx.get("own") is not first
    assert fiber.reloads == 1
    assert fiber.reload_reason == ("dep:impl",)


def test_failed_reload_rolls_back_partial_effects() -> None:
    """新实现 apply 失败：部分登记被撤销，状态 FAILED 且可观测。"""
    fail = {"on": False}

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide("own", object()))
        if fail["on"]:
            raise RuntimeError("new impl broken")

    ctx = Context()
    ctx.provide("dep", ImplV1())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert ctx.get("own") is not None

    fail["on"] = True
    ctx.provide("dep", ImplV2())

    assert fiber.state is FiberState.FAILED
    assert ctx.registered_keys() == ("dep",)  # own 已随失败回滚
    assert len(fiber._disposables) == 0
    assert fiber.rollbacks == 1
    assert isinstance(fiber.last_reload_error, RuntimeError)
    assert fiber.reload_reason == ("dep:impl",)


def test_retry_after_failed_reload_recovers() -> None:
    """FAILED 后显式 retry 可恢复，并计入 reloads。"""
    fail = {"on": True}

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide("own", object()))
        if fail["on"]:
            raise RuntimeError("transient")

    ctx = Context()
    ctx.provide("dep", object())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert fiber.state is FiberState.FAILED

    fail["on"] = False
    fiber.retry()

    assert fiber.state is FiberState.ACTIVE
    assert fiber.reloads == 1
    assert ctx.get("own") is not None


def test_config_change_reports_config_field() -> None:
    """仅注入配置变化：重载理由为 ``config`` 而非 ``impl``。"""
    spec = InjectSpec("dep", config={"mode": "a"})

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide("own", ctx.get("dep")))

    ctx = Context()
    ctx.provide("dep", object())
    fiber = ctx.plugin(Plugin("p", apply, inject=[spec]))
    assert fiber.reloads == 0

    spec.config["mode"] = "b"  # type: ignore[index]
    fiber._refresh()

    assert fiber.reloads == 1
    assert fiber.reload_reason == ("dep:config",)


def test_dependency_loss_retires_fiber_effects() -> None:
    """依赖消失：撤销 Fiber 效应并进入 INACTIVE，原因可解释。"""

    def apply(ctx, config) -> None:
        ctx.effect(lambda: ctx.provide("own", 1))

    ctx = Context()
    undo = ctx.provide("dep", object())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert ctx.get("own") == 1

    undo()

    assert fiber.state is FiberState.INACTIVE
    assert fiber.reason == ("dep",)
    assert "own" not in ctx.registered_keys()
    assert len(fiber._disposables) == 0


def test_consumer_hot_swap_preserves_downstream_key() -> None:
    """依赖方热替换后，其向下游提供的键仍指向新实现。"""
    produced: list[str] = []

    def apply(ctx, config) -> None:
        value = ctx.get("k")
        produced.append(value)  # type: ignore[arg-type]
        ctx.effect(lambda: ctx.provide("c", value))

    ctx = Context()
    ctx.provide("k", "v1")
    fiber = ctx.plugin(Plugin("consumer", apply, inject=[InjectSpec("k")]))
    assert ctx.get("c") == "v1"

    ctx.provide("k", "v2")

    assert fiber.reloads == 1
    assert produced == ["v1", "v2"]
    assert ctx.get("c") == "v2"
