"""装配与事务回滚测试：逆序回滚、异常聚合与静默断言（P1-2 验收）。"""

import pytest
from xuan_plugin.assembly import assemble
from xuan_plugin.manifest import InjectDecl, PluginManifest, ProvideDecl
from xuan_plugin.plugin import InjectSpec, Plugin
from xuan_plugin.solver import solve

KEY_A = "xuan:demo/a#one"
KEY_B = "xuan:demo/b#two"


def _manifest(plugin_id: str, *, provide: tuple[str, ...] = (), inject: tuple[str, ...] = ()) -> PluginManifest:
    return PluginManifest(
        id=plugin_id,
        version="0.1.0",
        provide=tuple(ProvideDecl(key=key) for key in provide),
        inject=tuple(InjectDecl(key=key) for key in inject),
    )


def _plugin(
    name: str,
    provide: tuple[str, ...] = (),
    inject: tuple[str, ...] = (),
    *,
    log: list[str] | None = None,
    register: bool = True,
    fail: bool = False,
    broken_undo: bool = False,
) -> Plugin:
    def apply(ctx, config) -> None:
        if log is not None:
            log.append(name)
        if register:
            for key in provide:
                ctx.provide(key, f"{name}:{key}")

        def undo() -> None:
            if broken_undo:
                raise RuntimeError(f"{name} 撤销失败")
            if log is not None:
                log.append(f"undo:{name}")

        ctx.effect(lambda: [undo], label=name)
        if fail:
            raise RuntimeError(f"{name} 装配失败")

    return Plugin(name=name, apply=apply, provide=list(provide), inject=[InjectSpec(name=key) for key in inject])


def test_assemble_applies_in_solution_order() -> None:
    log: list[str] = []
    manifests = [_manifest("provider", provide=(KEY_A,)), _manifest("consumer", provide=(KEY_B,), inject=(KEY_A,))]
    plugins = {
        "provider": _plugin("provider", (KEY_A,), log=log),
        "consumer": _plugin("consumer", (KEY_B,), (KEY_A,), log=log),
    }
    result = assemble(solve(manifests), plugins)

    assert result.ok
    assert result.applied == ("provider", "consumer")
    assert result.rolled_back == ()
    assert result.reconciled is not None and result.reconciled.ok
    assert result.reconciled.matched == (KEY_A, KEY_B)
    assert result.silenced
    assert log == ["provider", "consumer"]


def test_failure_rolls_back_in_reverse_and_leaves_no_residue() -> None:
    log: list[str] = []
    manifests = [_manifest("a", provide=(KEY_A,)), _manifest("b", provide=(KEY_B,), inject=(KEY_A,))]
    plugins = {
        "a": _plugin("a", (KEY_A,), log=log),
        "b": _plugin("b", (KEY_B,), (KEY_A,), log=log, fail=True),
    }
    result = assemble(solve(manifests), plugins)

    assert not result.ok
    assert [label for label, _ in result.errors] == ["apply:b"]
    assert result.rolled_back == ("a",)
    assert result.residual_effects == 0
    assert result.residual_keys == ()
    assert result.silenced
    assert result.reconciled is None
    # 已装配节点的撤销按逆序执行；失败节点自身的登记也一并回收
    assert log == ["a", "b", "undo:b", "undo:a"]


def test_unregistered_provide_leaves_consumer_inactive_and_fails() -> None:
    """供应者未真正登记键时消费者停留在 PENDING，不得被计为装配成功。"""
    manifests = [_manifest("a", provide=(KEY_A,)), _manifest("b", inject=(KEY_A,))]
    plugins = {"a": _plugin("a", (KEY_A,), register=False), "b": _plugin("b", inject=(KEY_A,))}
    result = assemble(solve(manifests), plugins)

    assert not result.ok
    assert [label for label, _ in result.errors] == ["apply:b"]
    assert "节点未激活" in str(result.errors[0][1])
    assert result.rolled_back == ("a",)


def test_dispose_errors_are_aggregated_but_residue_still_zero() -> None:
    manifests = [_manifest("a", provide=(KEY_A,)), _manifest("b", inject=(KEY_A,))]
    plugins = {"a": _plugin("a", (KEY_A,), broken_undo=True), "b": _plugin("b", fail=True)}
    result = assemble(solve(manifests), plugins)

    assert not result.ok
    labels = [label for label, _ in result.errors]
    assert "apply:b" in labels
    assert any(label.startswith("root:") for label in labels)
    assert result.residual_effects == 0
    assert result.silenced


def test_declared_but_unregistered_keys_are_reported_not_fatal() -> None:
    result = assemble(solve([_manifest("a", provide=(KEY_A,))]), {"a": _plugin("a", (KEY_A,), register=False)})

    assert result.ok
    assert result.reconciled is not None
    assert result.reconciled.missing == (("a", KEY_A),)
    assert result.reconciled.ok is False


def test_missing_implementation_is_a_caller_error() -> None:
    with pytest.raises(KeyError, match="缺少插件实现"):
        assemble(solve([_manifest("a")]), {})
