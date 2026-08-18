"""Harness 自举与插件装配的单元测试。"""

from pathlib import Path

from okf.harness import Harness
from okf.models import Bundle, ConformanceReport
from okf.plugin import InjectSpec, Plugin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _harness_for(path) -> Harness:
    """创建 Harness 并加载默认插件清单。"""
    harness = Harness(bundle_path=path)
    harness._load_default_plugins()
    return harness


def test_load_default_plugins_registers_expected_services(sample_bundle_path):
    harness = _harness_for(sample_bundle_path)

    expected = [
        "bundle_accessor",
        "conformance_report",
        "index_generator",
        "log_generator",
        "trust_analyzer",
        "link_analyzer",
    ]
    for name in expected:
        assert harness.ctx.get(name) is not None


def test_bundle_loader_provides_bundle_accessor(sample_bundle_path):
    harness = _harness_for(sample_bundle_path)

    bundle = harness.ctx.get("bundle_accessor")
    assert isinstance(bundle, Bundle)
    assert bundle.root == Path(sample_bundle_path)
    assert "concepts/metrics/revenue" in bundle.concepts


def test_conformance_checker_depends_on_bundle_accessor(sample_bundle_path):
    harness = _harness_for(sample_bundle_path)

    report = harness.ctx.get("conformance_report")
    assert isinstance(report, ConformanceReport)
    assert report.errors == []


def test_topological_sort_orders_dependencies():
    harness = Harness()

    loader = Plugin(name="loader", apply=lambda ctx, cfg: None, provide=["svc"])
    dep1 = Plugin(name="dep1", apply=lambda ctx, cfg: None, inject=[InjectSpec("svc")])
    dep2 = Plugin(name="dep2", apply=lambda ctx, cfg: None, inject=[InjectSpec("svc")])

    # 故意乱序传入，验证依赖排序结果不受声明顺序影响
    sorted_plugins = harness._topological_sort([dep2, loader, dep1])
    assert [p.name for p in sorted_plugins][0] == "loader"
    assert set(p.name for p in sorted_plugins) == {"loader", "dep1", "dep2"}


def test_from_config_reads_tool_okf_plugins(sample_bundle_path):
    harness = Harness.from_config(pyproject_path=PYPROJECT, bundle_path=sample_bundle_path)

    bundle = harness.ctx.get("bundle_accessor")
    assert isinstance(bundle, Bundle)
    report = harness.ctx.get("conformance_report")
    assert isinstance(report, ConformanceReport)


def test_from_config_loads_bundle_loader_first(sample_bundle_path):
    harness = Harness.from_config(pyproject_path=PYPROJECT, bundle_path=sample_bundle_path)

    names = list(harness.ctx._fibers.keys())
    assert names[0] == "bundle_loader"
    assert names.index("bundle_loader") < names.index("conformance_checker")
    assert names.index("bundle_loader") < names.index("trust_deriver")


def test_from_config_with_tmp_pyproject(tmp_path, sample_bundle_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.okf.plugins]\n"
        'bundle_loader = "okf.plugins.loader:BundleLoader"\n'
        'conformance_checker = "okf.plugins.conformance:ConformanceChecker"\n',
        encoding="utf-8",
    )

    harness = Harness.from_config(pyproject_path=pyproject, bundle_path=sample_bundle_path)

    assert isinstance(harness.ctx.get("bundle_accessor"), Bundle)
    assert isinstance(harness.ctx.get("conformance_report"), ConformanceReport)


def test_end_to_end_sample_bundle_services(sample_bundle_path):
    harness = _harness_for(sample_bundle_path)

    # conformance_report
    report = harness.ctx.get("conformance_report")
    assert report.errors == []

    # index_generator
    generator = harness.ctx.get("index_generator")
    content = generator()
    assert content.startswith("# sample_bundle")
    assert "concepts/metrics/revenue.md" in content

    # trust_analyzer
    analyzer = harness.ctx.get("trust_analyzer")
    revenue = analyzer("concepts/metrics/revenue")
    assert revenue["trust_tier"] == "human_reviewed"
    assert revenue["is_stale"] is True

    # link_analyzer
    link_analyzer = harness.ctx.get("link_analyzer")
    result = link_analyzer()
    assert result["broken"] == []
    assert len(result["links"]) == 1


# ── dispose / 上下文管理器 ────────────────────────────────────────────────


def test_harness_dispose_idempotent():
    harness = Harness()
    harness.dispose()
    harness.dispose()  # 重复调用幂等安全


def test_harness_context_manager():
    harness = Harness()
    harness.ctx.provide("svc", object())

    with harness as h:
        assert h is harness
        assert harness.ctx.get("svc") is not None

    # __exit__ 已 dispose，清空内部 Context
    assert harness.ctx._store == {}


# ── from_config 边界 ─────────────────────────────────────────────────────


def test_from_config_no_arg_walks_up_to_find_pyproject():
    """无参数调用向上查找 pyproject.toml 并返回 Harness。"""
    harness = Harness.from_config()
    assert isinstance(harness, Harness)


def test_from_config_nonexistent_pyproject_uses_defaults(tmp_path, sample_bundle_path):
    harness = Harness.from_config(
        pyproject_path=tmp_path / "nonexistent.toml", bundle_path=sample_bundle_path
    )
    assert isinstance(harness.ctx.get("bundle_accessor"), Bundle)
    assert isinstance(harness.ctx.get("conformance_report"), ConformanceReport)


def test_from_config_empty_plugin_map_uses_defaults(tmp_path, sample_bundle_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.okf]\nversion = '0.1'\n", encoding="utf-8")

    harness = Harness.from_config(pyproject_path=pyproject, bundle_path=sample_bundle_path)
    assert isinstance(harness.ctx.get("bundle_accessor"), Bundle)
    assert isinstance(harness.ctx.get("conformance_report"), ConformanceReport)


# ── 插件加载异常 / 非 Plugin 可调用对象 / 循环依赖 ─────────────────────────


def test_load_plugins_from_map_non_plugin_callable():
    harness = Harness()
    harness._load_plugins_from_map({"x": "okf.context:Context"})
    # 非 Plugin 的可调用对象被包装为 Plugin 并注册 fiber
    assert "x" in harness.ctx._fibers


def test_load_plugins_from_map_failure_prints_warning(capsys):
    harness = Harness()
    harness._load_plugins_from_map({"bad": "nonexistent_module:NoClass"})
    assert "Failed to load plugin" in capsys.readouterr().out


def test_topological_sort_cycle_fallback():
    harness = Harness()
    a = Plugin(name="a", apply=lambda c, cfg: None, inject=[InjectSpec("b_svc")], provide=["a_svc"])
    b = Plugin(name="b", apply=lambda c, cfg: None, inject=[InjectSpec("a_svc")], provide=["b_svc"])

    sorted_plugins = harness._topological_sort([a, b])
    assert set(p.name for p in sorted_plugins) == {"a", "b"}
