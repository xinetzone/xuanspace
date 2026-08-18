"""Capability Seam（ServiceDefinition / ServiceProvider / ServiceRegistry）测试。"""

from okf.context import Context
from okf.service import ServiceDefinition, ServiceProvider, ServiceRegistry


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


def test_registry_lookup_instantiates_first_provider():
    """ServiceRegistry.lookup 用第一个 provider 实例化。"""
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
    reg.register(
        ServiceProvider(d, lambda ctx, cfg: cfg["value"], config={"value": 99})
    )
    ctx = Context()
    assert reg.lookup("svc", ctx) == 99
    # 显式 config 覆盖 Provider config
    assert reg.lookup("svc", ctx, config={"value": 1}) == 1


def test_consumer_transparent_replacement():
    """Consumer 透明替换：切换 provider 后 lookup 返回新实现。"""
    reg = ServiceRegistry()
    d = ServiceDefinition("svc", object)
    reg.register(ServiceProvider(d, lambda ctx, cfg: "v1"))
    ctx = Context()
    assert reg.lookup("svc", ctx) == "v1"

    # 切换为新 provider（插入到最前，lookup 取第一个）
    providers = reg.get_providers("svc")
    providers.insert(0, ServiceProvider(d, lambda ctx, cfg: "v2"))
    assert reg.lookup("svc", ctx) == "v2"
