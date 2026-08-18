"""Capability Seam 三角色抽象：ServiceDefinition / ServiceProvider / ServiceRegistry。

仅使用 Python 标准库 ``dataclasses`` 与 ``collections.abc``，零第三方依赖。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context

__all__ = [
    "ServiceDefinition",
    "ServiceProvider",
    "ServiceRegistry",
]


@dataclass(frozen=True, slots=True)
class ServiceDefinition:
    """能力的接口契约，只声明'能做什么'，不定义'怎么做'。"""

    name: str
    interface: type
    description: str = ""


@dataclass(frozen=True, slots=True)
class ServiceProvider:
    """ServiceDefinition 的具体实现，可多个并存。"""

    definition: ServiceDefinition
    factory: Callable[[Context, dict], object]
    config: dict = field(default_factory=dict)


class ServiceRegistry:
    """服务注册中心，管理 ServiceDefinition 和 ServiceProvider 的注册与查找。"""

    def __init__(self) -> None:
        self._definitions: dict[str, ServiceDefinition] = {}
        self._providers: dict[str, list[ServiceProvider]] = {}

    def define(self, definition: ServiceDefinition) -> None:
        """注册 ServiceDefinition。同名覆盖，最新定义生效。"""
        self._definitions[definition.name] = definition

    def register(self, provider: ServiceProvider) -> None:
        """注册 ServiceProvider，追加到对应 name 的列表。"""
        name = provider.definition.name
        if name not in self._providers:
            self._providers[name] = []
        self._providers[name].append(provider)

    def get_definition(self, name: str) -> ServiceDefinition | None:
        """获取 ServiceDefinition，不存在时返回 None。"""
        return self._definitions.get(name)

    def get_providers(self, name: str) -> list[ServiceProvider]:
        """获取所有 ServiceProvider，不存在时返回空列表。"""
        return self._providers.get(name, [])

    def lookup(
        self, name: str, ctx: Context, config: dict | None = None
    ) -> object | None:
        """查找并实例化服务。使用第一个注册的 Provider 创建实例。

        Parameters
        ----------
        name:
            服务名称。
        ctx:
            请求上下文。
        config:
            可选配置字典，若为 None 则使用 Provider 自身的 config。

        Returns
        -------
        object or None
            实例化结果，无 Provider 时返回 None。
        """
        providers = self._providers.get(name)
        if not providers:
            return None
        provider = providers[0]
        effective_config = config if config is not None else provider.config
        return provider.factory(ctx, effective_config)
