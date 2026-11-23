"""Capability Seam 运行时裁决（P2-10）测试：ADR-05 多提供者定序三段规则。

规则来源（设计文档 ADR-05）：
1. ``multi`` 键按 ``priority`` 降序取最高者；
2. 消费者可显式指定 ``provider`` 覆盖 ``priority`` 结果；
3. ``priority`` 相同且未显式指定时报错——**不猜、不取注册顺序首个**。

``ServiceRegistry`` 当前无生产消费者，故此处以直接调用 ``lookup`` 验证语义。
"""

import pytest
from xuan_plugin.context import Context
from xuan_plugin.service import AmbiguousProviderError, ServiceDefinition, ServiceProvider, ServiceRegistry

_DEF = ServiceDefinition(name="xuan:demo/render#html", interface=object)


def _provider(pid: str = "", *, priority: int = 0, config: dict | None = None, result: str | None = None):
    """构造提供者；``result`` 缺省时以 ``pid`` 作为工厂返回值，便于断言选中的是谁。"""

    def factory(ctx: Context, cfg: dict) -> object:
        return (result if result is not None else pid, dict(cfg))

    return ServiceProvider(
        definition=_DEF,
        factory=factory,
        config={} if config is None else config,
        priority=priority,
        id=pid,
    )


def _registry(*providers: ServiceProvider) -> ServiceRegistry:
    registry = ServiceRegistry()
    registry.define(_DEF)
    for provider in providers:
        registry.register(provider)
    return registry


def test_lookup_without_provider_returns_none() -> None:
    """无提供者不是异常，沿用既有语义返回 None。"""
    registry = _registry()
    assert registry.lookup(_DEF.name, Context()) is None


def test_single_provider_selected() -> None:
    registry = _registry(_provider("only", config={"k": 1}))
    assert registry.lookup(_DEF.name, Context()) == ("only", {"k": 1})


@pytest.mark.parametrize("reverse", [False, True], ids=["registration-order", "reverse-order"])
def test_highest_priority_wins_regardless_of_registration_order(reverse: bool) -> None:
    """规则 1：取 priority 最高者，结果与注册顺序无关。"""
    low = _provider("low", priority=10)
    high = _provider("high", priority=200)
    providers = (high, low) if reverse else (low, high)
    assert _registry(*providers).lookup(_DEF.name, Context())[0] == "high"


def test_tied_priority_raises_instead_of_taking_first() -> None:
    """规则 3：最高优先级并列即失败，绝不隐式取注册顺序首个。"""
    registry = _registry(_provider("first", priority=50), _provider("second", priority=50))
    with pytest.raises(AmbiguousProviderError) as excinfo:
        registry.lookup(_DEF.name, Context())
    error = excinfo.value
    assert error.name == _DEF.name
    assert error.providers == ("first", "second")
    assert "无法裁决" in str(error)


def test_tied_priority_lower_levels_do_not_matter() -> None:
    """只有最高优先级并列才算歧义，低优先级同名者不参与。"""
    registry = _registry(
        _provider("top-a", priority=100),
        _provider("top-b", priority=100),
        _provider("low", priority=1),
    )
    with pytest.raises(AmbiguousProviderError) as excinfo:
        registry.lookup(_DEF.name, Context())
    assert excinfo.value.providers == ("top-a", "top-b")


def test_blank_id_falls_back_to_priority_label_in_error() -> None:
    """缺省 id 时错误信息以 priority 标注，便于诊断。"""
    registry = _registry(_provider("", priority=7), _provider("", priority=7))
    with pytest.raises(AmbiguousProviderError) as excinfo:
        registry.lookup(_DEF.name, Context())
    assert excinfo.value.providers == ("priority=7", "priority=7")


def test_explicit_provider_overrides_lower_priority() -> None:
    """规则 2：显式指定可覆盖 priority 裁决。"""
    registry = _registry(_provider("want", priority=1), _provider("default", priority=99))
    assert registry.lookup(_DEF.name, Context(), provider="want")[0] == "want"


def test_explicit_provider_resolves_tie() -> None:
    """规则 3 的出口：并列时由消费者显式指定即可继续。"""
    registry = _registry(_provider("a", priority=5), _provider("b", priority=5))
    assert registry.lookup(_DEF.name, Context(), provider="b")[0] == "b"


def test_explicit_provider_unknown_raises_lookup_error() -> None:
    registry = _registry(_provider("known", priority=5))
    with pytest.raises(LookupError) as excinfo:
        registry.lookup(_DEF.name, Context(), provider="missing")
    message = str(excinfo.value)
    assert "missing" in message
    assert "known" in message


def test_explicit_provider_duplicate_id_raises_ambiguous() -> None:
    """显式指定也无法裁决同 id 的两份实现，仍须报错而非取首个。"""
    registry = _registry(_provider("dup", priority=1), _provider("dup", priority=9))
    with pytest.raises(AmbiguousProviderError) as excinfo:
        registry.lookup(_DEF.name, Context(), provider="dup")
    assert excinfo.value.providers == ("dup", "dup")


def test_explicit_config_overrides_provider_config() -> None:
    registry = _registry(_provider("only", config={"source": "provider"}))
    assert registry.lookup(_DEF.name, Context(), {"source": "caller"})[1] == {"source": "caller"}


def test_get_definition_and_providers_accessors() -> None:
    registry = _registry(_provider("a"), _provider("b"))
    assert registry.get_definition(_DEF.name) is _DEF
    assert [p.id for p in registry.get_providers(_DEF.name)] == ["a", "b"]
    assert registry.get_providers("xuan:demo/missing#x") == []
