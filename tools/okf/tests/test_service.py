"""Capability Seam（ServiceDefinition / ServiceProvider / ServiceRegistry）测试。"""

from xuan_plugin.context import Context
from xuan_plugin.service import ServiceDefinition, ServiceProvider, ServiceRegistry


def test_service_definition_fields():
    """ServiceDefinition dataclass 字段（name/interface/description）。"""
    d = ServiceDefinition("svc", int, "desc")
    assert d.name == "svc"
    assert d.interface is int
    assert d.description == "desc"

    # description 默认为空
    assert ServiceDefinition("bare", int).description == ""


def test_registry_define_and_get_definition():
    """ServiceRegistry.define 注册接口契约，get_definition 查询。"""
    reg = ServiceRegistry()
    definition = ServiceDefinition("svc", int)
    reg.define(definition)
    assert reg.get_definition("svc") is definition
    assert reg.get_definition("missing") is None


def test_registry_register_and_get_providers():
    """ServiceRegistry.register 注册 provider，get_providers 查询。"""
    reg = ServiceRegistry()
    d = ServiceDefinition("svc", int)
    p = ServiceProvider(d, lambda ctx, cfg: 42)
    reg.register(p)
    assert reg.get_providers("svc") == [p]
    assert reg.get_providers("missing") == []


def test_registry_lookup_instantiates_selected_provider():
    """ServiceRegistry.lookup 用裁决出的 provider 实例化（此处仅一个候选）。"""
    reg = ServiceRegistry()
    d = ServiceDefinition("svc", int)
    reg.register(ServiceProvider(d, lambda ctx, cfg: 42))
    ctx = Context()
    assert reg.lookup("svc", ctx) == 42
    assert reg.lookup("missing", ctx) is None


def test_lookup_uses_provider_config_when_no_explicit_config():
    """未显式传 config 时使用 Provider 自身的 config。"""
    reg = ServiceRegistry()
    d = ServiceDefinition("svc", int)
    reg.register(ServiceProvider(d, lambda ctx, cfg: cfg["value"], config={"value": 99}))
    ctx = Context()
    assert reg.lookup("svc", ctx) == 99
    # 显式 config 覆盖 Provider config
    assert reg.lookup("svc", ctx, config={"value": 1}) == 1


def test_consumer_transparent_replacement():
    """Consumer 透明替换：新 provider 以更高 priority 成为默认，旧实现仍可显式指定。"""
    reg = ServiceRegistry()
    d = ServiceDefinition("svc", object)
    reg.register(ServiceProvider(d, lambda ctx, cfg: "v1", id="legacy"))
    ctx = Context()
    assert reg.lookup("svc", ctx) == "v1"

    # 追加新 provider 并抬高优先级：默认裁决透明切到新实现（ADR-05 规则 1）
    reg.register(ServiceProvider(d, lambda ctx, cfg: "v2", priority=1, id="next"))
    assert reg.lookup("svc", ctx) == "v2"
    # 消费者仍可回退到旧实现（ADR-05 规则 2）
    assert reg.lookup("svc", ctx, provider="legacy") == "v1"
