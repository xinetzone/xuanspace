"""插件清单（K2 契约）的解析、校验与装配图构建。

职责边界：本模块只做「静态契约」——把 TOML 中的 ``[tool.xuan.plugin]`` 解析为
不可变数据、校验字段与能力键语法、把多份清单合并成待求解的节点/边集合。
**不做**激活与卸载（``Fiber`` 负责），**不做**拓扑求解与回滚（P1-2 负责）。

字段与语法以设计文档 §5.2 K2、§5.3 清单示例为准。
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# 能力键语法：xuan:<domain>/<owner>#<api>
_KEY_RE = re.compile(r"^xuan:[a-z][a-z0-9_-]*/[a-z][a-z0-9_-]*#[a-z][a-z0-9_-]*$")

_MANIFEST_FIELDS = frozenset(
    {
        "id",
        "version",
        "provide",
        "inject",
        "isolate",
        "config_schema",
        "reversible",
        "priority",
        "disabled",
        "lifecycle",
    }
)
_PROVIDE_FIELDS = frozenset({"key", "mode"})
_INJECT_FIELDS = frozenset({"key", "optional", "constraint"})
_MODES = frozenset({"single", "multi"})
_LIFECYCLES = frozenset({"reloadable", "static"})


class ManifestError(ValueError):
    """清单不合约（K2 违约）。

    ``field`` 指出违约字段的路径（如 ``provide[2].key``），便于 ``xs plugin why``
    后续定位；消息面向使用者，因此使用中文并给出期望值。
    """

    def __init__(self, message: str, *, field: str = "") -> None:
        self.field = field
        prefix = f"[{field}] " if field else ""
        super().__init__(f"{prefix}{message}")


@dataclass(frozen=True, slots=True)
class ProvideDecl:
    """一条能力供应声明。"""

    key: str
    mode: str = "single"


@dataclass(frozen=True, slots=True)
class InjectDecl:
    """一条能力依赖声明。``constraint`` 描述结构约束（方法集与签名），非版本号。"""

    key: str
    optional: bool = False
    constraint: str = ""


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """单个插件的清单。

    ``source`` 不是清单字段，而是装配器的诊断标记（来源 pyproject 相对路径或
    entry point 描述），不参与契约比较。
    """

    id: str
    version: str
    provide: tuple[ProvideDecl, ...] = ()
    inject: tuple[InjectDecl, ...] = ()
    isolate: tuple[str, ...] = ()
    config_schema: Mapping[str, object] | None = field(default=None)
    reversible: bool = True
    priority: int = 0
    disabled: bool = False
    lifecycle: str = "reloadable"
    source: str = ""

    def contract(self) -> tuple:
        """契约视图：剔除 ``source`` 后的可比较元组，用于判定「同 id 是否同一契约」。"""
        return (
            self.id,
            self.version,
            self.provide,
            self.inject,
            self.isolate,
            self.reversible,
            self.priority,
            self.disabled,
            self.lifecycle,
        )

    def provide_keys(self) -> tuple[str, ...]:
        return tuple(p.key for p in self.provide)


@dataclass(frozen=True, slots=True)
class Edge:
    """依赖边：消费者 inject 的键指向某提供者。``provider`` 为 None 表示未解析。"""

    consumer: str
    key: str
    provider: str | None


@dataclass(frozen=True, slots=True)
class PluginGraph:
    """候选节点集与依赖边（求解前的形态）。"""

    nodes: tuple[str, ...]
    edges: tuple[Edge, ...]
    unresolved: tuple[tuple[str, str], ...]
    conflicts: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dot(self) -> str:
        """输出 Graphviz DOT；缺失依赖以虚线指向 ``MISSING`` 汇聚节点。"""
        lines = ["digraph xuan {", '  rankdir="LR";']
        for node in self.nodes:
            lines.append(f'  "{node}";')
        for edge in self.edges:
            if edge.provider is None:
                lines.append('  "MISSING" [shape=box, style=dashed];')
                lines.append(f'  "{edge.consumer}" -> "MISSING" [style=dashed, label="{edge.key}"];')
            else:
                lines.append(f'  "{edge.provider}" -> "{edge.consumer}" [label="{edge.key}"];')
        lines.append("}")
        return "\n".join(lines)


def _check_fields(data: Mapping[str, object], allowed: frozenset[str], ctx: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ManifestError(f"未知字段 {', '.join(unknown)}", field=ctx)


def _require_str(data: Mapping[str, object], name: str, ctx: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"必填字段 {name} 须为非空字符串", field=f"{ctx}.{name}" if ctx else name)
    return value.strip()


def _check_key(key: str, ctx: str) -> str:
    if not _KEY_RE.match(key):
        raise ManifestError(
            f"能力键 {key!r} 不合约，期望 xuan:<domain>/<owner>#<api>（小写字母开头，允许数字、下划线与连字符）",
            field=ctx,
        )
    return key


def _as_bool(data: Mapping[str, object], name: str, ctx: str, default: bool) -> bool:
    value = data.get(name, default)
    if not isinstance(value, bool):
        raise ManifestError(f"{name} 须为布尔值", field=f"{ctx}.{name}" if ctx else name)
    return value


def _as_int(data: Mapping[str, object], name: str, ctx: str, default: int) -> int:
    value = data.get(name, default)
    # TOML 的布尔是 int 子类，须先排除
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{name} 须为整数", field=f"{ctx}.{name}" if ctx else name)
    return value


def _as_str_tuple(data: Mapping[str, object], name: str, ctx: str) -> tuple[str, ...]:
    value = data.get(name, ())
    if not isinstance(value, (list, tuple)):
        raise ManifestError(f"{name} 须为字符串数组", field=f"{ctx}.{name}" if ctx else name)
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ManifestError("须为非空字符串", field=f"{ctx}.{name}[{index}]")
        result.append(item.strip())
    return tuple(result)


def _parse_provide(raw: object, ctx: str) -> tuple[ProvideDecl, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError("provide 须为表数组（[[tool.xuan.plugin.provide]]）", field=f"{ctx}.provide")
    decls: list[ProvideDecl] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"{ctx}.provide[{index}]"
        if not isinstance(item, Mapping):
            raise ManifestError("每项须为表", field=path)
        _check_fields(item, _PROVIDE_FIELDS, path)
        key = _check_key(_require_str(item, "key", path), f"{path}.key")
        mode = item.get("mode", "single")
        if not isinstance(mode, str) or mode not in _MODES:
            raise ManifestError(f"mode 须为 {sorted(_MODES)} 之一", field=f"{path}.mode")
        if key in seen:
            raise ManifestError(f"provide 键重复：{key}", field=f"{path}.key")
        seen.add(key)
        decls.append(ProvideDecl(key=key, mode=mode))
    return tuple(decls)


def _parse_inject(raw: object, ctx: str) -> tuple[InjectDecl, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ManifestError("inject 须为表数组（[[tool.xuan.plugin.inject]]）", field=f"{ctx}.inject")
    decls: list[InjectDecl] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"{ctx}.inject[{index}]"
        if not isinstance(item, Mapping):
            raise ManifestError("每项须为表", field=path)
        _check_fields(item, _INJECT_FIELDS, path)
        key = _check_key(_require_str(item, "key", path), f"{path}.key")
        optional = _as_bool(item, "optional", path, False)
        constraint = item.get("constraint", "")
        if not isinstance(constraint, str):
            raise ManifestError("constraint 须为字符串", field=f"{path}.constraint")
        if key in seen:
            raise ManifestError(f"inject 键重复：{key}", field=f"{path}.key")
        seen.add(key)
        decls.append(InjectDecl(key=key, optional=optional, constraint=constraint))
    return tuple(decls)


def parse_manifest(data: Mapping[str, object], *, source: str = "") -> PluginManifest:
    """解析并校验单份清单（``[tool.xuan.plugin]`` 的表内容）。"""
    if not isinstance(data, Mapping):
        raise ManifestError("清单须为表")
    _check_fields(data, _MANIFEST_FIELDS, source or "[tool.xuan.plugin]")

    plugin_id = _require_str(data, "id", "")
    version = _require_str(data, "version", "")
    lifecycle = data.get("lifecycle", "reloadable")
    if not isinstance(lifecycle, str) or lifecycle not in _LIFECYCLES:
        raise ManifestError(f"lifecycle 须为 {sorted(_LIFECYCLES)} 之一", field="lifecycle")

    config_schema = data.get("config_schema")
    if config_schema is not None and not isinstance(config_schema, Mapping):
        raise ManifestError("config_schema 须为表", field="config_schema")

    return PluginManifest(
        id=plugin_id,
        version=version,
        provide=_parse_provide(data.get("provide"), plugin_id),
        inject=_parse_inject(data.get("inject"), plugin_id),
        isolate=_as_str_tuple(data, "isolate", plugin_id),
        config_schema=config_schema,
        reversible=_as_bool(data, "reversible", plugin_id, True),
        priority=_as_int(data, "priority", plugin_id, 0),
        disabled=_as_bool(data, "disabled", plugin_id, False),
        lifecycle=lifecycle,
        source=source,
    )


def manifest_from_pyproject(data: Mapping[str, object], *, source: str = "") -> PluginManifest | None:
    """从整份 ``pyproject.toml`` 数据中取出清单；没有 ``[tool.xuan.plugin]`` 时返回 None。"""
    tool = data.get("tool")
    if not isinstance(tool, Mapping):
        return None
    xuan = tool.get("xuan")
    if not isinstance(xuan, Mapping):
        return None
    plugin = xuan.get("plugin")
    if plugin is None:
        return None
    return parse_manifest(plugin, source=source)


def merge_manifests(manifests: Iterable[PluginManifest]) -> tuple[PluginManifest, ...]:
    """按 id 合并，返回按 id 排序的清单元组。

    同一 id 出现两份**不同契约**（``source`` 不计）时报错——这正对应 K2 的
    「清单/实现双向一致」，不允许同 id 双真相；两份完全相同时保留先到者。
    """
    merged: dict[str, PluginManifest] = {}
    for manifest in manifests:
        existing = merged.get(manifest.id)
        if existing is None:
            merged[manifest.id] = manifest
        elif existing.contract() != manifest.contract():
            raise ManifestError(
                f"插件 id {manifest.id!r} 出现两份不同契约：{existing.source or '未知来源'} 与 {manifest.source or '未知来源'}",
                field=f"{manifest.id}.id",
            )
    return tuple(merged[key] for key in sorted(merged))


def _providers_by_key(manifests: Iterable[PluginManifest]) -> dict[str, list[tuple[str, str, int]]]:
    """key → [(plugin_id, mode, priority)]，按 priority 降序、id 升序稳定定序。"""
    index: dict[str, list[tuple[str, str, int]]] = {}
    for manifest in manifests:
        if manifest.disabled:
            continue
        for decl in manifest.provide:
            index.setdefault(decl.key, []).append((manifest.id, decl.mode, manifest.priority))
    for providers in index.values():
        providers.sort(key=lambda item: (-item[2], item[0]))
    return index


def build_graph(manifests: Iterable[PluginManifest]) -> PluginGraph:
    """把清单集合展开为节点、依赖边、未解析依赖与冲突键。

    * 节点 = 未被 ``disabled`` 的插件 id（字典序）；
    * 边 = 每个 inject 指向该键的提供者（多提供者时取 priority 最高者，见 ADR-05）；
    * 冲突 = 同一键被多个插件声明为 ``single``（``multi`` 显式声明者免冲突，K5）。
    """
    active = [m for m in manifests if not m.disabled]
    index = _providers_by_key(active)

    conflicts: list[tuple[str, tuple[str, ...]]] = []
    for key, providers in sorted(index.items()):
        if len(providers) < 2:
            continue
        if any(mode == "multi" for _, mode, _ in providers):
            continue
        conflicts.append((key, tuple(plugin_id for plugin_id, _, _ in providers)))

    edges: list[Edge] = []
    unresolved: list[tuple[str, str]] = []
    for manifest in sorted(active, key=lambda m: m.id):
        for decl in manifest.inject:
            providers = index.get(decl.key, [])
            provider = providers[0][0] if providers else None
            edges.append(Edge(consumer=manifest.id, key=decl.key, provider=provider))
            if provider is None and not decl.optional:
                unresolved.append((manifest.id, decl.key))

    return PluginGraph(
        nodes=tuple(sorted(m.id for m in active)),
        edges=tuple(edges),
        unresolved=tuple(unresolved),
        conflicts=tuple(conflicts),
    )
