"""插件清单（K2）解析、合并与装配图构建测试。"""

import pytest
from xuan_plugin.manifest import (
    Edge,
    InjectDecl,
    ManifestError,
    ProvideDecl,
    build_graph,
    manifest_from_pyproject,
    merge_manifests,
    parse_manifest,
)


def _minimal(**overrides) -> dict:
    data: dict = {"id": "demo", "version": "0.1.0"}
    data.update(overrides)
    return data


def test_parse_minimal_defaults() -> None:
    """只给 id 与 version 时其余字段取默认值。"""
    manifest = parse_manifest(_minimal())
    assert manifest.id == "demo"
    assert manifest.version == "0.1.0"
    assert manifest.provide == ()
    assert manifest.inject == ()
    assert manifest.isolate == ()
    assert manifest.reversible is True
    assert manifest.priority == 0
    assert manifest.disabled is False
    assert manifest.lifecycle == "reloadable"


def test_parse_full_manifest() -> None:
    data = _minimal(
        priority=100,
        reversible=False,
        lifecycle="static",
        isolate=["env:workspace"],
        provide=[{"key": "xuan:cli/commands#list", "mode": "single"}],
        inject=[{"key": "xuan:build/backend#dispatch", "optional": True, "constraint": "methods:build"}],
    )
    manifest = parse_manifest(data, source="tools/xs/pyproject.toml")
    assert manifest.priority == 100
    assert manifest.reversible is False
    assert manifest.lifecycle == "static"
    assert manifest.isolate == ("env:workspace",)
    assert manifest.provide == (ProvideDecl(key="xuan:cli/commands#list", mode="single"),)
    assert manifest.inject == (
        InjectDecl(key="xuan:build/backend#dispatch", optional=True, constraint="methods:build"),
    )
    assert manifest.source == "tools/xs/pyproject.toml"


@pytest.mark.parametrize("missing", ["id", "version"])
def test_required_fields(missing: str) -> None:
    data = _minimal()
    del data[missing]
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(data)
    assert missing in str(excinfo.value)


def test_invalid_key_syntax() -> None:
    """能力键必须符合 xuan:<domain>/<owner>#<api>。"""
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_minimal(provide=[{"key": "cli/list"}]))
    assert "能力键" in str(excinfo.value)


def test_invalid_mode() -> None:
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_minimal(provide=[{"key": "xuan:cli/commands#list", "mode": "shared"}]))
    assert "mode" in str(excinfo.value)


def test_invalid_lifecycle() -> None:
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_minimal(lifecycle="hot"))
    assert "lifecycle" in str(excinfo.value)


def test_unknown_field_rejected() -> None:
    """拼错的字段必须被拦截，而非静默忽略。"""
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_minimal(priorty=10))
    assert "未知字段" in str(excinfo.value)


def test_duplicate_provide_key() -> None:
    data = _minimal(provide=[{"key": "xuan:cli/commands#list"}, {"key": "xuan:cli/commands#list"}])
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(data)
    assert "重复" in str(excinfo.value)


def test_bool_is_not_int_for_priority() -> None:
    """TOML 布尔是 int 子类，priority=true 必须被拒绝。"""
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_minimal(priority=True))
    assert "priority" in str(excinfo.value)


def test_manifest_from_pyproject_absent() -> None:
    assert manifest_from_pyproject({"project": {"name": "demo"}}) is None
    assert manifest_from_pyproject({"tool": {"xuan": {}}}) is None


def test_manifest_from_pyproject_extracts() -> None:
    data = {"tool": {"xuan": {"plugin": _minimal(priority=5)}}}
    manifest = manifest_from_pyproject(data, source="demo/pyproject.toml")
    assert manifest is not None
    assert manifest.priority == 5
    assert manifest.source == "demo/pyproject.toml"


def test_merge_manifests_dedupes_identical() -> None:
    a = parse_manifest(_minimal(), source="a/pyproject.toml")
    b = parse_manifest(_minimal(), source="b/pyproject.toml")
    merged = merge_manifests([b, a])
    assert len(merged) == 1
    assert merged[0].source == "b/pyproject.toml"


def test_merge_manifests_rejects_divergent_contracts() -> None:
    a = parse_manifest(_minimal(version="0.1.0"), source="a")
    b = parse_manifest(_minimal(version="0.2.0"), source="b")
    with pytest.raises(ManifestError) as excinfo:
        merge_manifests([a, b])
    assert "不同契约" in str(excinfo.value)


def _provider(plugin_id: str, key: str, *, mode: str = "single", priority: int = 0, disabled: bool = False):
    return parse_manifest(
        _minimal(
            id=plugin_id,
            provide=[{"key": key, "mode": mode}],
            priority=priority,
            disabled=disabled,
        )
    )


def test_build_graph_edges_and_priority_choice() -> None:
    """多提供者时按 priority 降序取最高者（ADR-05）。"""
    high = _provider("high", "xuan:cli/commands#list", mode="multi", priority=200)
    low = _provider("low", "xuan:cli/commands#list", mode="multi", priority=10)
    consumer = parse_manifest(_minimal(id="consumer", inject=[{"key": "xuan:cli/commands#list"}]))
    graph = build_graph([low, high, consumer])
    assert graph.nodes == ("consumer", "high", "low")
    assert graph.edges == (Edge(consumer="consumer", key="xuan:cli/commands#list", provider="high"),)
    assert graph.conflicts == ()


def test_build_graph_optional_inject_not_unresolved() -> None:
    manifest = parse_manifest(_minimal(inject=[{"key": "xuan:doc/render#html", "optional": True}]))
    graph = build_graph([manifest])
    assert graph.unresolved == ()
    assert graph.edges[0].provider is None


def test_build_graph_required_inject_unresolved() -> None:
    manifest = parse_manifest(_minimal(inject=[{"key": "xuan:doc/render#html"}]))
    graph = build_graph([manifest])
    assert graph.unresolved == (("demo", "xuan:doc/render#html"),)


def test_build_graph_single_conflict_and_multi_exempt() -> None:
    a = _provider("a", "xuan:cli/commands#list")
    b = _provider("b", "xuan:cli/commands#list")
    graph = build_graph([a, b])
    assert graph.conflicts == (("xuan:cli/commands#list", ("a", "b")),)

    a_multi = _provider("a", "xuan:cli/commands#list", mode="multi")
    multi_graph = build_graph([a_multi, b])
    assert multi_graph.conflicts == ()


def test_build_graph_skips_disabled_nodes() -> None:
    disabled = _provider("off", "xuan:cli/commands#list", disabled=True)
    graph = build_graph([disabled])
    assert graph.nodes == ()
    assert graph.edges == ()


def test_graph_dot_output() -> None:
    provider = _provider("core", "xuan:doc/render#html")
    consumer = parse_manifest(_minimal(id="consumer", inject=[{"key": "xuan:doc/render#html"}]))
    dot = build_graph([provider, consumer]).to_dot()
    assert dot.startswith("digraph xuan {")
    assert '"core" -> "consumer" [label="xuan:doc/render#html"];' in dot


def test_manifest_contract_ignores_source() -> None:
    a = parse_manifest(_minimal(), source="a")
    b = parse_manifest(_minimal(), source="b")
    assert a.contract() == b.contract()
    assert a != b
