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
        ctx.provide("own", 1)
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
        ctx.provide("own", impl)

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
        ctx.provide("own", object())
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
        ctx.provide("own", object())
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
        ctx.provide("own", ctx.get("dep"))

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
        ctx.provide("own", 1)

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
        ctx.provide("c", value)

    ctx = Context()
    ctx.provide("k", "v1")
    fiber = ctx.plugin(Plugin("consumer", apply, inject=[InjectSpec("k")]))
    assert ctx.get("c") == "v1"

    ctx.provide("k", "v2")

    assert fiber.reloads == 1
    assert produced == ["v1", "v2"]
    assert ctx.get("c") == "v2"


def test_reload_converges_to_inactive_when_inputs_vanish_during_apply() -> None:
    """apply 期间依赖消失：`_reload` 自行收敛到 INACTIVE，不依赖后续通知补救。"""
    ctx = Context()
    undo = ctx.provide("dep", ImplV1())

    def apply(ctx, config) -> None:
        ctx.get("dep")
        undo()  # 撤销供应；notify 被外层事务延后，排空不会及时补救

    with ctx.batch():
        fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))

        assert fiber.state is FiberState.INACTIVE
        assert fiber.reason == ("dep",)
        assert len(fiber._disposables) == 0

    # 事务排空后仍稳定在 INACTIVE
    assert fiber.state is FiberState.INACTIVE
    assert fiber.reason == ("dep",)


def test_hot_reload_with_vanishing_dependency_is_neither_reload_nor_rollback() -> None:
    """热替换的新装配期间依赖消失：撤销效应、收敛 INACTIVE，且不计成功也不计回滚。"""
    undone: list[str] = []
    calls = {"n": 0}
    undo_new: list = []

    def apply(ctx, config) -> None:
        calls["n"] += 1
        ctx.effect(lambda: [lambda: undone.append("undo")])
        if calls["n"] == 2:
            undo_new[0]()  # 新实现装配期间依赖被撤销

    ctx = Context()
    ctx.provide("dep", ImplV1())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    assert fiber.state is FiberState.ACTIVE

    with ctx.batch():
        undo_new.append(ctx.provide("dep", ImplV2()))
        fiber._refresh()  # 事务内触发「先卸旧、再装新」

        assert fiber.state is FiberState.INACTIVE
        assert fiber.reason == ("dep",)
        assert fiber.reloads == 0
        assert fiber.rollbacks == 0
        assert fiber.pending_reload is None
        assert len(fiber._disposables) == 0
        assert undone == ["undo", "undo"]

    assert fiber.state is FiberState.INACTIVE


def test_bare_provide_is_registered_as_fiber_effect() -> None:
    """apply 中直接 provide（不再包 effect）也应归属 Fiber，卸载后键被回收（P2-12）。"""

    def apply(ctx, config) -> None:
        ctx.provide("own", 1)

    ctx = Context()
    fiber = ctx.plugin(Plugin("p", apply))
    assert fiber.state is FiberState.ACTIVE
    assert ctx.get("own") == 1
    assert len(fiber._disposables) == 1
    assert len(ctx._root_disposables) == 0

    fiber._unload()

    assert "own" not in ctx.registered_keys()
    assert len(fiber._disposables) == 0


def test_bare_provide_is_rolled_back_when_apply_fails() -> None:
    """apply 中途失败：裸 provide 写入的键必须随 Fiber 回滚，不留半成品。"""

    def apply(ctx, config) -> None:
        ctx.provide("own", 1)
        raise RuntimeError("boom")

    ctx = Context()
    fiber = ctx.plugin(Plugin("p", apply))

    assert fiber.state is FiberState.FAILED
    assert ctx.registered_keys() == ()
    assert len(fiber._disposables) == 0


def test_bare_provide_follows_hot_swap() -> None:
    """热替换：旧值随旧效应撤销，新装配写入新值，下游读到的是当前实现。"""
    versions: list[object] = []

    def apply(ctx, config) -> None:
        value = ctx.get("dep")
        versions.append(value)
        ctx.provide("own", value)

    ctx = Context()
    ctx.provide("dep", ImplV1())
    fiber = ctx.plugin(Plugin("p", apply, inject=[InjectSpec("dep")]))
    first = versions[0]
    assert ctx.get("own") is first

    ctx.provide("dep", ImplV2())

    assert fiber.reloads == 1
    assert ctx.get("own") is versions[1]
    assert ctx.get("own") is not first


def test_repeated_provide_keeps_one_registration_per_key() -> None:
    """同名重复供应只保留最后一次撤销登记，避免长期运行下登记数线性增长。"""
    ctx = Context()
    ctx.provide("k", "v1")
    ctx.provide("k", "v2")

    assert ctx.get("k") == "v2"
    assert len(ctx._root_disposables) == 1

    ctx.dispose()

    assert ctx.registered_keys() == ()
    assert len(ctx._root_disposables) == 0


def test_provide_label_is_recorded_as_effect_meta() -> None:
    """provide 的 label 进入效应观测树，与 effect(label=...) 同口径。"""
    ctx = Context()
    ctx.provide("k", "v", label="provide:k")

    assert [meta.label for meta in ctx.get_effects()] == ["provide:k"]
