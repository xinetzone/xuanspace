"""Xuanspace 插件内核。

从 ``tools/okf`` 抽取的 Cordis 组件范式实现，提供可撤销效果、反应式协效应、
能力注册与插件生命周期状态机。本包零运行时依赖，仅使用标准库。
"""

from .context import Context
from .disposable import Disposable, DisposableList, DisposalError, EffectMeta
from .plugin import Fiber, FiberState, InjectSpec, Plugin
from .service import ServiceDefinition, ServiceProvider, ServiceRegistry

__all__ = [
    "Context",
    "Disposable",
    "DisposableList",
    "DisposalError",
    "EffectMeta",
    "Fiber",
    "FiberState",
    "InjectSpec",
    "Plugin",
    "ServiceDefinition",
    "ServiceProvider",
    "ServiceRegistry",
]
