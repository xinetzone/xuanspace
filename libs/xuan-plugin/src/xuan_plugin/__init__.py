"""Xuanspace 插件内核。

从 ``tools/okf`` 抽取的 Cordis 组件范式实现，提供可撤销效果、反应式协效应、
能力注册与插件生命周期状态机。本包零运行时依赖，仅使用标准库。
"""

from .assembly import AssemblyResult, assemble
from .context import Context
from .disposable import Disposable, DisposableList, DisposalError, EffectMeta
from .manifest import (
    Edge,
    InjectDecl,
    ManifestError,
    PluginGraph,
    PluginManifest,
    ProvideDecl,
    build_graph,
    manifest_from_pyproject,
    merge_manifests,
    parse_manifest,
)
from .plugin import Fiber, FiberState, FiberTransitionError, InjectSpec, Plugin
from .service import AmbiguousProviderError, ServiceDefinition, ServiceProvider, ServiceRegistry
from .solver import (
    Domain,
    InactiveReason,
    ReconcileReport,
    Solution,
    SolveError,
    SolveErrorKind,
    reconcile,
    solve,
)

__all__ = [
    "AmbiguousProviderError",
    "AssemblyResult",
    "Context",
    "Disposable",
    "DisposableList",
    "DisposalError",
    "Domain",
    "Edge",
    "EffectMeta",
    "Fiber",
    "FiberState",
    "FiberTransitionError",
    "InactiveReason",
    "InjectDecl",
    "InjectSpec",
    "ManifestError",
    "Plugin",
    "PluginGraph",
    "PluginManifest",
    "ProvideDecl",
    "ReconcileReport",
    "ServiceDefinition",
    "ServiceProvider",
    "ServiceRegistry",
    "Solution",
    "SolveError",
    "SolveErrorKind",
    "assemble",
    "build_graph",
    "manifest_from_pyproject",
    "merge_manifests",
    "parse_manifest",
    "reconcile",
    "solve",
]
