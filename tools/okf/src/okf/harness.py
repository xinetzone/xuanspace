"""Harness 自举 —— 从 pyproject.toml 配置驱动插件装配。"""

from __future__ import annotations

import importlib
import tomllib
import traceback
from collections import deque
from pathlib import Path

from .context import Context
from .plugin import Plugin


class Harness:
    """OKF 工具链 Harness，从 pyproject.toml 配置驱动插件装配。

    设计原则（DeepSeek Harness 启发）：
    - 无特权内核：所有插件地位平等
    - 一切皆插件：所有能力都是可插拔插件
    - 配置驱动：插件装配由 pyproject.toml 驱动
    - 可替换性：任意插件可被第三方实现替换
    """

    def __init__(self, bundle_path: Path | str | None = None) -> None:
        self._ctx = Context()
        self._bundle_path = Path(bundle_path) if bundle_path else None

    @property
    def ctx(self) -> Context:
        return self._ctx

    def dispose(self) -> None:
        """释放 Harness：清理内部 Context 的所有资源，幂等安全。"""
        self._ctx.dispose()

    def __enter__(self) -> Harness:
        """进入上下文：退出时自动回收资源（contextlib 语义落地）。"""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文：自动 ``dispose()``。"""
        self.dispose()

    @classmethod
    def from_config(
        cls,
        pyproject_path: Path | str | None = None,
        bundle_path: Path | str | None = None,
    ) -> Harness:
        """从 pyproject.toml 的 [tool.okf.plugins] 配置节创建 Harness。

        自举流程：
        1. 读取 pyproject.toml 的 [tool.okf.plugins] 配置节
        2. 创建 Context 根实例
        3. 按拓扑排序（依赖声明的 inject 约束）依次加载插件
        4. 每个插件加载成功后，其 provide 的服务自动通知 notify
        5. 所有插件 ACTIVE 后，Harness 就绪
        """
        harness = cls(bundle_path=bundle_path)

        if pyproject_path is None:
            # 从当前文件向上查找 pyproject.toml
            current = Path(__file__).resolve().parent
            while current != current.parent:
                candidate = current / "pyproject.toml"
                if candidate.exists():
                    pyproject_path = candidate
                    break
                current = current.parent

        if pyproject_path is None or not Path(pyproject_path).exists():
            # 无 pyproject.toml，使用默认插件清单
            harness._load_default_plugins()
            return harness

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        plugin_map = config.get("tool", {}).get("okf", {}).get("plugins", {})
        if not plugin_map:
            harness._load_default_plugins()
            return harness

        harness._load_plugins_from_map(plugin_map)
        return harness

    def _plugin_config(self, name: str) -> dict | None:
        """为 bundle_loader 插件注入 bundle 路径配置。"""
        if name == "bundle_loader" and self._bundle_path is not None:
            return {"path": str(self._bundle_path)}
        return None

    def _load_plugins_from_map(self, plugin_map: dict[str, str]) -> None:
        """从插件名→导入路径映射加载插件。"""
        plugins: list[Plugin] = []
        for name, import_path in plugin_map.items():
            try:
                module_path, class_name = import_path.rsplit(":", 1)
                module = importlib.import_module(module_path)
                plugin_cls = getattr(module, class_name)
                if callable(plugin_cls):
                    plugin_instance = plugin_cls()
                    if isinstance(plugin_instance, Plugin):
                        plugins.append(plugin_instance)
                    else:
                        plugins.append(Plugin(name=name, apply=plugin_instance))
            except Exception:
                print(f"Warning: Failed to load plugin '{name}' from '{import_path}':")
                print(traceback.format_exc())

        # 拓扑排序：按依赖关系排序
        sorted_plugins = self._topological_sort(plugins)
        for plugin in sorted_plugins:
            self._ctx.plugin(plugin, self._plugin_config(plugin.name))

    def _load_default_plugins(self) -> None:
        """加载默认插件清单。"""
        from .plugins.cli_adapter import CLIAdapter
        from .plugins.conformance import ConformanceChecker
        from .plugins.links import LinkResolver
        from .plugins.loader import BundleLoader
        from .plugins.synthesis import IndexSynthesizer, LogSynthesizer
        from .plugins.trust import TrustDeriver

        # 1. bundle_loader（无依赖，先加载）
        self._ctx.plugin(BundleLoader(), self._plugin_config("bundle_loader"))

        # 2. 依赖 bundle_accessor 的插件
        for plugin_cls in [ConformanceChecker, IndexSynthesizer, LogSynthesizer, TrustDeriver, LinkResolver]:
            self._ctx.plugin(plugin_cls())

        # 3. CLI 适配层（依赖全部服务，最后加载）
        self._ctx.plugin(CLIAdapter())

    def _topological_sort(self, plugins: list[Plugin]) -> list[Plugin]:
        """按拓扑排序插件（基于 inject 依赖声明）。"""
        name_to_plugin = {p.name: p for p in plugins}
        provides = {}
        for p in plugins:
            for svc in p.provide:
                provides[svc] = p.name

        in_degree: dict[str, int] = {}
        adj: dict[str, list[str]] = {}
        for p in plugins:
            if p.name not in in_degree:
                in_degree[p.name] = 0
            if p.name not in adj:
                adj[p.name] = []
            for spec in p.inject:
                if spec.name in provides:
                    dep_name = provides[spec.name]
                    if dep_name not in adj:
                        adj[dep_name] = []
                    if dep_name not in in_degree:
                        in_degree[dep_name] = 0
                    adj.setdefault(dep_name, []).append(p.name)
                    in_degree[p.name] = in_degree.get(p.name, 0) + 1

        # Kahn's algorithm
        queue = deque(name for name in in_degree if in_degree[name] == 0)
        result: list[Plugin] = []
        while queue:
            name = queue.popleft()
            if name in name_to_plugin:
                result.append(name_to_plugin[name])
            for neighbor in adj.get(name, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 添加未被拓扑排序覆盖的插件（循环依赖等情况）
        for p in plugins:
            if p not in result:
                result.append(p)

        return result
