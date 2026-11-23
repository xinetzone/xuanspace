"""Capability Seam 三角色抽象：ServiceDefinition / ServiceProvider / ServiceRegistry。

仅使用 Python 标准库 ``dataclasses`` 与 ``collections.abc``，零第三方依赖。
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context

__all__ = [
    "AmbiguousProviderError",
    "ServiceDefinition",
    "ServiceProvider",
    "ServiceRegistry",
]


class AmbiguousProviderError(RuntimeError):
    """同优先级的多个提供者之间无法裁决（ADR-05 第 3 条）。

    不退回「取首个」：注册顺序不是语义身份，同一组提供者换个加载顺序就会
    得到不同实现，与「最终配置与插入顺序无关」直接冲突。歧义在解析点显式
    失败，由调用方补 ``priority`` 或显式指定 ``provider``。
    """

    def __init__(self, name: str, providers: tuple[str, ...]) -> None:
        self.name = name
        self.providers = providers
        super().__init__(f"服务 {name} 的提供者优先级相同，无法裁决：{', '.join(providers)}")


@dataclass(frozen=True, slots=True)
class ServiceDefinition:
    """能力的接口契约，只声明'能做什么'，不定义'怎么做'。"""

    name: str
    interface: type
    description: str = ""


@dataclass(frozen=True, slots=True)
class ServiceProvider:
    """ServiceDefinition 的具体实现，可多个并存。

    ``priority`` 与清单 K2 的同名字段同口径：数值大者胜出。``id`` 是供显式
    覆盖使用的稳定标识（对应 ADR-05 第 2 条「消费者可强制指定」），缺省为空
    串时只参与优先级裁决。
    """

    definition: ServiceDefinition
    factory: Callable[[Context, dict], object]
    config: dict = field(default_factory=dict)
    priority: int = 0
    id: str = ""


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
        self,
        name: str,
        ctx: Context,
        config: dict | None = None,
        *,
        provider: str | None = None,
    ) -> object | None:
        """按 ADR-05 三段规则选择并实例化服务。

        1. 默认取 ``priority`` 最高者；最高值并列时抛
           :class:`AmbiguousProviderError`——既不猜、也不取注册顺序首个；
        2. 显式给出 ``provider`` 时以该 ``id`` 强制指定，覆盖 priority 结果；
        3. 无任何提供者时返回 ``None``（「依赖不可得不是异常」的既有语义）。

        Parameters
        ----------
        name:
            服务名称。
        ctx:
            请求上下文。
        config:
            可选配置字典，若为 None 则使用被选中 Provider 自身的 config。
        provider:
            可选 Provider ``id``，用于显式覆盖优先级裁决。

        Returns
        -------
        object or None
            实例化结果，无 Provider 时返回 None。
        """
        providers = self._providers.get(name)
        if not providers:
            return None
        chosen = self._select(name, providers, provider)
        effective_config = config if config is not None else chosen.config
        return chosen.factory(ctx, effective_config)

    @staticmethod
    def _select(name: str, providers: list[ServiceProvider], provider: str | None) -> ServiceProvider:
        """按 ADR-05 裁决单个提供者；歧义即失败，不做隐式兜底。"""
        if provider is not None:
            matched = [p for p in providers if p.id == provider]
            if not matched:
                known = ", ".join(sorted({p.id for p in providers}))
                raise LookupError(f"服务 {name} 不存在 id 为 {provider!r} 的提供者；已注册：{known}")
            if len(matched) > 1:
                raise AmbiguousProviderError(name, tuple(provider for _ in matched))
            return matched[0]

        top = max(p.priority for p in providers)
        candidates = [p for p in providers if p.priority == top]
        if len(candidates) > 1:
            raise AmbiguousProviderError(
                name,
                tuple(p.id or f"priority={p.priority}" for p in candidates),
            )
        return candidates[0]
