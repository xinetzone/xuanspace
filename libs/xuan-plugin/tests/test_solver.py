"""装配求解器测试：拓扑定序、隔离域、可解性诊断与对账（P1-2 验收）。"""

import pytest
from xuan_plugin.manifest import InjectDecl, PluginManifest, ProvideDecl
from xuan_plugin.solver import (
    Domain,
    InactiveReason,
    SolveError,
    SolveErrorKind,
    reconcile,
    solve,
)

KEY_A = "xuan:demo/a#one"
KEY_B = "xuan:demo/b#two"
KEY_C = "xuan:demo/c#three"
KEY_D = "xuan:demo/d#four"


def _manifest(
    plugin_id: str,
    *,
    provide: tuple[str, ...] = (),
    inject: tuple[InjectDecl, ...] = (),
    isolate: tuple[str, ...] = (),
    priority: int = 0,
    disabled: bool = False,
) -> PluginManifest:
    return PluginManifest(
        id=plugin_id,
        version="0.1.0",
        provide=tuple(ProvideDecl(key=key) for key in provide),
        inject=inject,
        isolate=isolate,
        priority=priority,
        disabled=disabled,
    )


def _requires(*keys: str) -> tuple[InjectDecl, ...]:
    return tuple(InjectDecl(key=key) for key in keys)


def test_provider_precedes_consumer() -> None:
    manifests = [_manifest("consumer", inject=_requires(KEY_A)), _manifest("provider", provide=(KEY_A,))]
    solution = solve(manifests)
    assert solution.order == ("provider", "consumer")
    assert solution.inactive == ()


def test_ready_set_sorted_by_priority_then_id() -> None:
    manifests = [
        _manifest("b", priority=0),
        _manifest("a", priority=0),
        _manifest("c", priority=10),
    ]
    assert solve(manifests).order == ("c", "a", "b")


def test_order_is_independent_of_input_order() -> None:
    manifests = [
        _manifest("app", inject=_requires(KEY_A, KEY_B)),
        _manifest("aa", provide=(KEY_A,)),
        _manifest("bb", provide=(KEY_B,)),
    ]
    forward = solve(manifests)
    backward = solve(list(reversed(manifests)))
    assert forward == backward
    assert forward.order == ("aa", "bb", "app")


def test_self_dependency_is_peer_semantics() -> None:
    """同 id 内 inject 自己的 provide 键按 peer 语义可满足，不构成环。"""
    solution = solve([_manifest("solo", provide=(KEY_A,), inject=_requires(KEY_A))])
    assert solution.order == ("solo",)


def test_cross_cycle_is_rejected_with_deterministic_path() -> None:
    manifests = [
        _manifest("a", provide=(KEY_A,), inject=_requires(KEY_B)),
        _manifest("b", provide=(KEY_B,), inject=_requires(KEY_A)),
    ]
    with pytest.raises(SolveError) as info:
        solve(manifests)
    assert info.value.kind is SolveErrorKind.CYCLE
    assert "a -> b -> a" in str(info.value)
    # 同输入两次求解必须给出逐字相同的错误文本
    with pytest.raises(SolveError) as again:
        solve(list(reversed(manifests)))
    assert str(again.value) == str(info.value)


def test_single_key_conflict_is_rejected() -> None:
    manifests = [_manifest("a", provide=(KEY_A,)), _manifest("b", provide=(KEY_A,))]
    with pytest.raises(SolveError) as info:
        solve(manifests)
    assert info.value.kind is SolveErrorKind.CONFLICT
    assert info.value.details == (f"{KEY_A}: a, b",)


def test_multi_providers_require_distinct_priority() -> None:
    def multi(plugin_id: str, priority: int) -> PluginManifest:
        return PluginManifest(
            id=plugin_id,
            version="0.1.0",
            provide=(ProvideDecl(key=KEY_A, mode="multi"),),
            priority=priority,
        )

    with pytest.raises(SolveError) as info:
        solve([multi("a", 5), multi("b", 5)])
    assert info.value.kind is SolveErrorKind.CONFLICT

    solution = solve([multi("a", 10), multi("b", 0)])
    assert solution.order == ("a", "b")
    assert solution.provides == ((KEY_A, "a"),)


def test_missing_required_dependency_marks_inactive() -> None:
    solution = solve([_manifest("a", inject=_requires(KEY_A))])
    assert solution.order == ()
    assert solution.inactive == (InactiveReason(plugin="a", missing=(KEY_A,)),)


def test_inactive_propagates_to_dependents() -> None:
    manifests = [
        _manifest("a", inject=_requires(KEY_B)),
        _manifest("b", provide=(KEY_B,), inject=_requires(KEY_A)),
    ]
    solution = solve(manifests)
    assert solution.order == ()
    assert [reason.plugin for reason in solution.inactive] == ["a", "b"]


def test_optional_dependency_never_blocks() -> None:
    solution = solve([_manifest("a", inject=(InjectDecl(key=KEY_A, optional=True),))])
    assert solution.order == ("a",)
    assert solution.inactive == ()


def test_strict_mode_rejects_missing_dependencies() -> None:
    with pytest.raises(SolveError) as info:
        solve([_manifest("a", inject=_requires(KEY_A))], allow_inactive=False)
    assert info.value.kind is SolveErrorKind.MISSING
    assert info.value.details == (f"a 缺少 {KEY_A}",)


def test_disabled_manifest_is_excluded() -> None:
    manifests = [
        _manifest("a", inject=_requires(KEY_A)),
        _manifest("off", provide=(KEY_A,), disabled=True),
    ]
    solution = solve(manifests)
    assert solution.order == ()
    assert [reason.plugin for reason in solution.inactive] == ["a"]


def test_domains_cluster_by_shared_isolate_label() -> None:
    manifests = [
        _manifest("a", isolate=("env:workspace",)),
        _manifest("b", isolate=("env:workspace",)),
        _manifest("c", isolate=("env:npu",)),
        _manifest("d"),
    ]
    assert solve(manifests).domains == (
        Domain(labels=(), nodes=("d",)),
        Domain(labels=("env:npu",), nodes=("c",)),
        Domain(labels=("env:workspace",), nodes=("a", "b")),
    )


def test_domains_merge_transitively_across_labels() -> None:
    manifests = [
        _manifest("a", isolate=("env:one",)),
        _manifest("b", isolate=("env:one", "env:two")),
        _manifest("c", isolate=("env:two",)),
    ]
    assert solve(manifests).domains == (Domain(labels=("env:one", "env:two"), nodes=("a", "b", "c")),)


def test_provides_excludes_keys_of_inactive_nodes() -> None:
    manifests = [
        _manifest("a", provide=(KEY_A,), inject=_requires(KEY_C)),
        _manifest("b", provide=(KEY_B,)),
    ]
    solution = solve(manifests)
    assert solution.order == ("b",)
    assert solution.provides == ((KEY_B, "b"),)


def test_reconcile_reports_missing_and_unexpected() -> None:
    solution = solve([_manifest("a", provide=(KEY_A, KEY_B))])
    report = reconcile(solution, [KEY_A, "xuan:demo/z#extra"])
    assert report.missing == (("a", KEY_B),)
    assert report.unexpected == ("xuan:demo/z#extra",)
    assert report.matched == (KEY_A,)
    assert report.ok is False

    assert reconcile(solution, [KEY_A, KEY_B]).ok is True
