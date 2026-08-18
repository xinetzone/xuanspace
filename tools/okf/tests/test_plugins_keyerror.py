"""插件 KeyError 防御分支与生成器边界测试。

这些插件在依赖服务未注册时应安全返回（而非抛出异常），通过直接调用
其 ``__call__``（经由 ``Plugin.apply``）来覆盖防御分支。
"""

from datetime import date

import pytest

from okf.context import Context
from okf.models import Bundle
from okf.plugins.cli_adapter import CLIAdapter
from okf.plugins.conformance import ConformanceChecker
from okf.plugins.links import LinkResolver
from okf.plugins.synthesis import IndexSynthesizer, LogSynthesizer
from okf.plugins.trust import TrustDeriver


def _instance(plugin_cls):
    """返回插件的 __call__ 实例（Plugin.apply 字段）。"""
    return plugin_cls().apply


# ── KeyError 防御分支：依赖服务未注册时安全返回 ───────────────────────────


def test_cli_adapter_missing_services():
    ctx = Context()
    _instance(CLIAdapter)(ctx, {})
    names = [
        "bundle_accessor",
        "conformance_report",
        "index_generator",
        "log_generator",
        "trust_analyzer",
        "link_analyzer",
    ]
    assert ctx.get("cli_services") == {name: None for name in names}


def test_conformance_checker_missing_bundle():
    ctx = Context()
    _instance(ConformanceChecker)(ctx, {})
    with pytest.raises(KeyError):
        ctx.get("conformance_report")


def test_link_resolver_missing_bundle():
    ctx = Context()
    _instance(LinkResolver)(ctx, {})
    with pytest.raises(KeyError):
        ctx.get("link_analyzer")


def test_index_synthesizer_missing_bundle():
    ctx = Context()
    _instance(IndexSynthesizer)(ctx, {})
    with pytest.raises(KeyError):
        ctx.get("index_generator")


def test_log_synthesizer_missing_bundle():
    ctx = Context()
    _instance(LogSynthesizer)(ctx, {})
    with pytest.raises(KeyError):
        ctx.get("log_generator")


def test_trust_deriver_missing_bundle():
    ctx = Context()
    _instance(TrustDeriver)(ctx, {})
    with pytest.raises(KeyError):
        ctx.get("trust_analyzer")


# ── LogSynthesizer 生成器边界 ─────────────────────────────────────────────


def test_log_synthesizer_generate_with_log(tmp_path):
    (tmp_path / "log.md").write_text(
        "# Change Log\n\n## 2024-01-01\n- **Creation** x\n", encoding="utf-8"
    )
    ctx = Context()
    ctx.provide("bundle_accessor", Bundle(root=tmp_path))
    _instance(LogSynthesizer)(ctx, {})

    assert date(2024, 1, 1) in ctx.get("log_generator")()


def test_log_synthesizer_generate_without_log(tmp_path):
    ctx = Context()
    ctx.provide("bundle_accessor", Bundle(root=tmp_path))
    _instance(LogSynthesizer)(ctx, {})

    assert ctx.get("log_generator")() == {}


# ── TrustDeriver analyze 概念不存在 ───────────────────────────────────────


def test_trust_deriver_analyze_unknown_concept(tmp_path):
    ctx = Context()
    ctx.provide("bundle_accessor", Bundle(root=tmp_path))
    _instance(TrustDeriver)(ctx, {})

    assert ctx.get("trust_analyzer")("nonexistent") is None
