"""``xs plugin`` 命令：清单解析、装配图、求解与运行时装配（P1-1 / P1-2 / P1-3）。

数据来源为「静态声明 ∪ entry points」：前者扫描工作区内各项目 ``pyproject.toml``
的 ``[tool.xuan.plugin]``，后者读取 ``xuan.plugins`` 入口点。``apply --dry-run``
输出求解计划（定序/隔离域/可解性）；不加 ``--dry-run`` 时对内置插件执行真实
装配，并以「声明 / 实际注册」逐键对账收尾。``why`` 回答单个插件为何生效或
未生效——缺哪个键、谁可能提供、失活沿哪条依赖链传播（设计文档 V4）。
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from xuan_build import PLUGIN as BUILD_PLUGIN
from xuan_ffi import PLUGIN as FFI_PLUGIN
from xuan_plugin.assembly import assemble
from xuan_plugin.manifest import (
    PluginGraph,
    PluginManifest,
    build_graph,
    manifest_from_pyproject,
    merge_manifests,
)
from xuan_plugin.plugin import Plugin
from xuan_plugin.solver import Solution, SolveError, solve
from xuan_project import discover_projects, find_workspace_root, load_pyproject

from ..plugins.commands import PLUGIN as COMMANDS_PLUGIN
from ..plugins.registry import CommandRegistry

console = Console()

app = typer.Typer(
    help="插件清单与装配图（K1/K2 契约）",
    add_completion=False,
    no_args_is_help=True,
)

ENTRY_POINT_GROUP = "xuan.plugins"


def _collect_static(root: Path) -> list[PluginManifest]:
    """扫描工作区各项目的 pyproject.toml，收集静态清单。"""
    manifests: list[PluginManifest] = []
    for project in sorted(discover_projects(root), key=lambda item: item.name):
        if project.pyproject_path is None:
            continue
        try:
            source = project.pyproject_path.relative_to(root).as_posix()
        except ValueError:
            source = str(project.pyproject_path)
        manifest = manifest_from_pyproject(load_pyproject(project.pyproject_path), source=source)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def _collect_entry_points() -> list[PluginManifest]:
    """读取 ``xuan.plugins`` 入口点。

    入口点加载失败只跳过该项并告警——第三方插件损坏不应让整条命令失效。
    加载对象若暴露 ``MANIFEST``/``manifest`` 则直接采用，否则按其元数据合成
    最小清单，保证入口点至少在图中可见。
    """
    manifests: list[PluginManifest] = []
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            loaded = entry_point.load()
        except Exception as exc:  # noqa: BLE001 —— 第三方加载失败不得中断汇总
            console.print(f"[yellow]跳过入口点 {entry_point.name}: {exc}[/yellow]")
            continue
        manifest = getattr(loaded, "MANIFEST", None) or getattr(loaded, "manifest", None)
        if isinstance(manifest, PluginManifest):
            manifests.append(manifest)
            continue
        distribution = getattr(entry_point, "dist", None)
        manifests.append(
            PluginManifest(
                id=entry_point.name,
                version=getattr(distribution, "version", None) or "0.0.0",
                source=f"entry-point:{entry_point.value}",
            )
        )
    return manifests


def collect_manifests(root: Path) -> tuple[PluginManifest, ...]:
    """合并静态声明与入口点，返回按 id 排序的清单。"""
    return merge_manifests([*_collect_static(root), *_collect_entry_points()])


@app.command("list")
def list_plugins(
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出"),
) -> None:
    """列出工作区内声明的插件清单。"""
    try:
        manifests = collect_manifests(find_workspace_root())
    except Exception as exc:  # noqa: BLE001 —— 清单违约须给出可读错误而非栈回溯
        console.print(f"[red]清单解析失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    if json_output:
        payload = [
            {
                "id": m.id,
                "version": m.version,
                "lifecycle": m.lifecycle,
                "priority": m.priority,
                "provide": [{"key": p.key, "mode": p.mode} for p in m.provide],
                "inject": [{"key": i.key, "optional": i.optional} for i in m.inject],
                "source": m.source,
            }
            for m in manifests
        ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not manifests:
        console.print("[yellow]工作区内未发现 [tool.xuan.plugin] 声明，也无 xuan.plugins 入口点。[/yellow]")
        return

    # rich 默认在窄终端把单元格截断为省略号（如 "xs.comma…"）。处理方式：
    # 表头取紧凑写法、短字段禁换行、来源列折行、来源只显示项目目录，
    # 使 7 列在 80 列终端内完整放下且不丢信息。
    table = Table(title="插件清单", show_lines=False)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("VER", style="green", no_wrap=True)
    table.add_column("LIFECYCLE", no_wrap=True)
    table.add_column("PRIO", justify="right", no_wrap=True)
    table.add_column("PROV", justify="right", no_wrap=True)
    table.add_column("INJ", justify="right", no_wrap=True)
    table.add_column("SOURCE", style="dim", overflow="fold")
    for manifest in manifests:
        # 来源只展示所属项目目录——文件名恒为 pyproject.toml，完整路径由 --json 提供
        source = manifest.source
        if source.endswith("/pyproject.toml"):
            source = source[: -len("/pyproject.toml")]
        table.add_row(
            manifest.id,
            manifest.version,
            manifest.lifecycle,
            str(manifest.priority),
            str(len(manifest.provide)),
            str(len(manifest.inject)),
            source or "-",
        )
    console.print(table)
    keys = {p.key for m in manifests for p in m.provide}
    console.print(f"[dim]共 {len(manifests)} 个插件，{len(keys)} 个能力键[/dim]")


def _graph_diagnostics(graph_data: PluginGraph) -> tuple[str, ...]:
    """图诊断摘要：冲突与必需未解析共用一份文案，供文本/JSON/DOT 三种输出复用。

    只收「会让装配不可解」的问题——冲突键无唯一胜出者、必需依赖无任何提供者；
    可选依赖缺失不计入，避免把合法的可选降级当成错误。
    """
    lines: list[str] = [f"能力键冲突：{key}（{', '.join(providers)}）" for key, providers in graph_data.conflicts]
    lines.extend(f"未解析的必需依赖：{consumer} 需要 {key}" for consumer, key in graph_data.unresolved)
    return tuple(lines)


@app.command("graph")
def graph(
    dot: bool = typer.Option(False, "--dot", help="输出 Graphviz DOT"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出"),
) -> None:
    """输出候选节点与依赖边，并标注未解析依赖与冲突键。

    存在「能力键冲突」或「必需依赖未解析」时以退出码 1 结束：诊断图照常输出，
    便于 CI 与脚本拦截；文本/JSON/DOT 三种输出携带同一份诊断摘要。
    """
    try:
        manifests = collect_manifests(find_workspace_root())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]清单解析失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    graph_data = build_graph(manifests)
    diagnostics = _graph_diagnostics(graph_data)

    if dot:
        typer.echo(graph_data.to_dot())
        if diagnostics:
            # DOT 支持 ``//`` 注释：摘要追加在图后，图仍可被 Graphviz 正常渲染
            typer.echo(f"// 诊断摘要（{len(diagnostics)} 项，退出码 1）")
            for line in diagnostics:
                typer.echo(f"// {line}")
    elif json_output:
        typer.echo(
            json.dumps(
                {
                    "status": "error" if diagnostics else "ok",
                    "nodes": list(graph_data.nodes),
                    "edges": [{"consumer": e.consumer, "key": e.key, "provider": e.provider} for e in graph_data.edges],
                    "unresolved": [{"consumer": c, "key": k} for c, k in graph_data.unresolved],
                    "conflicts": [{"key": k, "providers": list(p)} for k, p in graph_data.conflicts],
                    "diagnostics": list(diagnostics),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        key_count = len({p.key for m in manifests for p in m.provide})
        console.print(
            f"[dim]插件 {len(graph_data.nodes)} 个，能力键 {key_count} 个，"
            f"依赖边 {len(graph_data.edges)} 条，未解析 {len(graph_data.unresolved)}，"
            f"冲突 {len(graph_data.conflicts)}[/dim]"
        )

        by_id = {m.id: m for m in manifests}
        conflicted_keys = {key for key, _ in graph_data.conflicts}
        unresolved_keys = {key for _, key in graph_data.unresolved}
        for node in graph_data.nodes:
            manifest = by_id[node]
            console.print(
                f"\n[bold cyan]{node}[/bold cyan] [dim]({manifest.lifecycle}, priority={manifest.priority},"
                f" 来源 {manifest.source or '未知'})[/dim]"
            )
            for provide in manifest.provide:
                console.print(f"  [green]provide[/green] {provide.key} [dim]({provide.mode})[/dim]")
            for edge in (e for e in graph_data.edges if e.consumer == node):
                if edge.provider is None:
                    if edge.key in conflicted_keys:
                        console.print(f"  [red]inject[/red] {edge.key} [dim]→ 冲突（无唯一胜出者）[/dim]")
                    else:
                        style = "red" if edge.key in unresolved_keys else "yellow"
                        console.print(f"  [{style}]inject[/{style}] {edge.key} [dim]→ 未解析[/dim]")
                else:
                    console.print(f"  inject {edge.key} [dim]→ {edge.provider}[/dim]")

        if diagnostics:
            console.print(f"\n[red]诊断摘要（{len(diagnostics)} 项，退出码 1）：[/red]")
            for line in diagnostics:
                console.print(f"  [red]{line}[/red]")

    if diagnostics:
        raise typer.Exit(1)


@app.command("apply")
def apply_plugins(
    dry_run: bool = typer.Option(False, "--dry-run", help="只输出求解计划，不执行装配"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出"),
) -> None:
    """求解并装配插件：定序、隔离域、可解性诊断与逐键对账。

    ``--dry-run`` 只输出求解计划；否则对已具备运行时实现的内置插件执行装配。
    缺少实现的插件会显式列名退出，不会把「什么都没做」伪装成装配成功。
    """
    try:
        manifests = collect_manifests(find_workspace_root())
        solution = solve(manifests)
    except SolveError as exc:
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "status": "error",
                        "kind": exc.kind.value,
                        "message": str(exc),
                        "details": list(exc.details),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            console.print(f"[red]装配计划无解：{exc}[/red]")
            for detail in exc.details:
                console.print(f"  {detail}")
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001 —— 清单违约须给出可读错误而非栈回溯
        console.print(f"[red]清单解析失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    if dry_run:
        _emit_plan(solution, manifests, json_output)
        return

    _apply_solution(solution, json_output)


def _emit_plan(solution: Solution, manifests: tuple[PluginManifest, ...], json_output: bool) -> None:
    """输出求解计划（``--dry-run``）。"""
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "status": "ok",
                    "order": list(solution.order),
                    "domains": [
                        {"labels": list(domain.labels), "nodes": list(domain.nodes)} for domain in solution.domains
                    ],
                    "inactive": [
                        {"plugin": reason.plugin, "missing": list(reason.missing)} for reason in solution.inactive
                    ],
                    "provides": [{"key": key, "provider": provider} for key, provider in solution.provides],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    by_id = {manifest.id: manifest for manifest in manifests}
    console.print(f"[bold]装配顺序[/bold] [dim]{len(solution.order)} 个节点[/dim]")
    for index, node in enumerate(solution.order, start=1):
        manifest = by_id[node]
        console.print(
            f"  {index:>2}. [cyan]{node}[/cyan] "
            f"[dim]priority={manifest.priority} provide={len(manifest.provide)} inject={len(manifest.inject)}[/dim]"
        )

    console.print("\n[bold]隔离域[/bold]")
    for domain in solution.domains:
        labels = ", ".join(domain.labels) or "（默认域）"
        console.print(f"  [magenta]{labels}[/magenta]：{', '.join(domain.nodes)}")

    if solution.inactive:
        console.print("\n[yellow]未激活（必需依赖不可得）：[/yellow]")
        for reason in solution.inactive:
            console.print(f"  {reason.describe()}")
    else:
        console.print("\n[dim]可解性：全部必需依赖可满足[/dim]")


def _apply_solution(solution: Solution, json_output: bool) -> None:
    """对内置插件执行真实装配，并输出对账结果。"""
    builtin: dict[str, Plugin] = {
        BUILD_PLUGIN.name: BUILD_PLUGIN,
        COMMANDS_PLUGIN.name: COMMANDS_PLUGIN,
        FFI_PLUGIN.name: FFI_PLUGIN,
    }
    missing = [node for node in solution.order if node not in builtin]
    if missing:
        message = f"缺少运行时实现：{', '.join(missing)}"
        if json_output:
            typer.echo(json.dumps({"status": "error", "kind": "missing_impl", "message": message}, ensure_ascii=False))
        else:
            console.print(f"[red]{message}[/red]")
        raise typer.Exit(1)

    registry = CommandRegistry()
    result = assemble(
        solution,
        {node: builtin[node] for node in solution.order},
        configs={COMMANDS_PLUGIN.name: {"registry": registry}},
    )
    if not result.ok:
        errors = [(label, f"{type(exc).__name__}: {exc}") for label, exc in result.errors]
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "status": "error",
                        "kind": "assembly",
                        "applied": list(result.applied),
                        "rolled_back": list(result.rolled_back),
                        "errors": [{"stage": label, "message": detail} for label, detail in errors],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            console.print("[red]装配失败，已回滚：[/red]")
            for label, detail in errors:
                console.print(f"  {label}: {detail}")
        raise typer.Exit(1)

    report = result.reconciled
    if report is None:
        raise RuntimeError("装配成功但缺少对账报告")
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "status": "ok",
                    "applied": list(result.applied),
                    "commands": [entry.name for entry in registry.entries()],
                    "reconciled": {
                        "missing": [{"provider": owner, "key": key} for owner, key in report.missing],
                        "unexpected": list(report.unexpected),
                        "matched": list(report.matched),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    console.print(
        f"[green]装配完成[/green]：{len(result.applied)} 个节点 ACTIVE，{len(registry.entries())} 条命令已登记"
    )
    console.print(
        f"[dim]对账：一致 {len(report.matched)} 键，缺 {len(report.missing)}，多 {len(report.unexpected)}[/dim]"
    )
    if not report.ok:
        raise typer.Exit(1)


def _key_owners(active: tuple[PluginManifest, ...]) -> dict[str, tuple[str, ...]]:
    """键 → 提供者 id（按 ``-priority``、id 定序）；与求解器选定胜出者同口径。"""
    ordered: dict[str, list[tuple[int, str]]] = {}
    for manifest in active:
        for decl in manifest.provide:
            ordered.setdefault(decl.key, []).append((-manifest.priority, manifest.id))
    return {key: tuple(plugin_id for _, plugin_id in sorted(items)) for key, items in ordered.items()}


@dataclass(frozen=True, slots=True)
class MissingKey:
    """一个不可得的键：候选提供者、被禁用来源，以及提供者失活时的下钻链。

    ``plugin`` 是「谁在缺这个键」——顶层为被查询插件，链上每一层为其提供者；
    这样同一条链既能渲染成人类可读的因果句，也能原样序列化为 JSON。
    """

    plugin: str
    key: str
    providers: tuple[tuple[str, str], ...]
    disabled_sources: tuple[str, ...]
    chain: tuple[MissingKey, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin,
            "key": self.key,
            "providers": [{"id": plugin_id, "state": state} for plugin_id, state in self.providers],
            "disabled_sources": list(self.disabled_sources),
            "chain": [item.to_json() for item in self.chain],
        }


def _missing_key(
    plugin: str,
    key: str,
    *,
    manifests: tuple[PluginManifest, ...],
    owners: Mapping[str, tuple[str, ...]],
    inactive: Mapping[str, tuple[str, ...]],
    depth: int = 0,
) -> MissingKey:
    """解释单个缺失键；提供者自身失活时继续下钻，直到找到根因。

    ``depth`` 是纯防御：``_inactive_nodes`` 的标记顺序本身保证失活链无环，
    但外部传入的 ``inactive`` 映射未必可信，故限深 8 层避免无限递归。
    """
    providers = owners.get(key, ())
    disabled_sources = tuple(
        manifest.id for manifest in manifests if manifest.disabled and any(decl.key == key for decl in manifest.provide)
    )
    provider = providers[0] if providers else ""
    chain: tuple[MissingKey, ...] = ()
    if provider and provider in inactive and depth < 8:
        chain = tuple(
            _missing_key(
                provider,
                nested,
                manifests=manifests,
                owners=owners,
                inactive=inactive,
                depth=depth + 1,
            )
            for nested in inactive[provider]
        )
    return MissingKey(
        plugin=plugin,
        key=key,
        providers=tuple((plugin_id, "inactive" if plugin_id in inactive else "active") for plugin_id in providers),
        disabled_sources=disabled_sources,
        chain=chain,
    )


def _provider_text(providers: tuple[tuple[str, str], ...]) -> str:
    """提供者列表的可读文案；空表示工作区内无任何声明。"""
    if not providers:
        return "无提供者"
    return "、".join(
        f"{plugin_id}（{'未生效' if state == 'inactive' else '已生效'}）" for plugin_id, state in providers
    )


def _render_chain(entry: MissingKey, indent: int) -> list[str]:
    """把下钻链渲染成逐层因果句（每层一行，缩进递增）。"""
    pad = " " * indent
    lines: list[str] = []
    for step in entry.chain:
        lines.append(f"{pad}[dim]因果：{step.plugin} 缺少 {step.key} → {_provider_text(step.providers)}[/dim]")
        if step.disabled_sources:
            lines.append(f"{pad}  [yellow]已禁用来源：{'、'.join(step.disabled_sources)}[/yellow]")
        lines.extend(_render_chain(step, indent + 2))
    return lines


def _render_missing(entry: MissingKey) -> list[str]:
    """渲染一个顶层缺失键：键与候选、禁用来源、下钻链与修复建议。"""
    lines = [f"  [red]{entry.key}[/red] [dim]→ {_provider_text(entry.providers)}[/dim]"]
    if entry.disabled_sources:
        lines.append(f"    [yellow]已禁用来源：{'、'.join(entry.disabled_sources)}[/yellow]")
        lines.append("    [dim]建议：把上述来源的 disabled 置为 false，或新增其它提供者[/dim]")
    elif not entry.providers:
        lines.append("    [dim]建议：新增提供该键的插件，或把该 inject 声明为 optional[/dim]")
    elif entry.chain:
        lines.append(f"    [dim]建议：先修复提供者 {entry.chain[0].plugin} 的缺失依赖[/dim]")
    lines.extend(_render_chain(entry, 4))
    return lines


@app.command("why")
def why(
    plugin_id: str = typer.Argument(..., metavar="PLUGIN_ID", help="要解释的插件 id"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出"),
) -> None:
    """解释插件为何生效/未生效：缺失键、可能提供者与失活因果链（V4）。

    退出码 0 仅表示「被查询插件已生效」；未生效（INACTIVE）、被禁用、未知 id
    与装配计划无解均为 1。这是「该插件当前能否生效」的问答语义，与 ``graph``/
    ``apply`` 的「装配计划是否可解」不同——INACTIVE 在求解层面是合法结果，
    ``apply --dry-run`` 遇到它仍返回 0。
    """
    try:
        manifests = collect_manifests(find_workspace_root())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]清单解析失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    target = next((manifest for manifest in manifests if manifest.id == plugin_id), None)
    if target is None:
        available = [manifest.id for manifest in manifests]
        if json_output:
            typer.echo(
                json.dumps({"status": "unknown", "id": plugin_id, "available": available}, ensure_ascii=False, indent=2)
            )
        else:
            console.print(f"[red]未找到插件 {plugin_id!r}[/red]")
            console.print(f"[dim]工作区内的插件：{', '.join(available) or '（无）'}[/dim]")
        raise typer.Exit(1)

    try:
        solution = solve(manifests)
    except SolveError as exc:
        if json_output:
            typer.echo(
                json.dumps(
                    {
                        "status": "blocked",
                        "id": plugin_id,
                        "kind": exc.kind.value,
                        "message": str(exc),
                        "details": list(exc.details),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            console.print(f"[red]装配计划无解，无法解释 {plugin_id}：{exc}[/red]")
            console.print(f"[dim]修复上述冲突/环后可重试：xs plugin why {plugin_id}[/dim]")
        raise typer.Exit(1)

    active = tuple(manifest for manifest in manifests if not manifest.disabled)
    owners = _key_owners(active)
    inactive = {reason.plugin: reason.missing for reason in solution.inactive}

    if target.disabled:
        status = "disabled"
    elif target.id in solution.order:
        status = "active"
    else:
        status = "inactive"

    missing = (
        tuple(
            _missing_key(target.id, key, manifests=manifests, owners=owners, inactive=inactive)
            for key in inactive.get(target.id, ())
        )
        if status == "inactive"
        else ()
    )
    position = solution.order.index(target.id) + 1 if status == "active" else None

    injects: list[dict[str, Any]] = []
    for decl in target.inject:
        providers = owners.get(decl.key, ())
        if not providers:
            state = "optional-missing" if decl.optional else "missing"
        elif providers[0] in inactive:
            state = "inactive-provider"
        else:
            state = "ok"
        injects.append(
            {
                "key": decl.key,
                "optional": decl.optional,
                "provider": providers[0] if providers else None,
                "state": state,
            }
        )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "status": status,
                    "id": target.id,
                    "manifest": {
                        "version": target.version,
                        "lifecycle": target.lifecycle,
                        "priority": target.priority,
                        "reversible": target.reversible,
                        "source": target.source,
                    },
                    "position": position,
                    "order": list(solution.order),
                    "provides": list(target.provide_keys()),
                    "injects": injects,
                    "missing": [entry.to_json() for entry in missing],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise typer.Exit(0 if status == "active" else 1)

    console.print(
        f"[bold cyan]{target.id}[/bold cyan] [dim]（{target.version}, {target.lifecycle},"
        f" priority={target.priority}, 来源 {target.source or '未知'}）[/dim]"
    )
    if status == "active":
        console.print(
            f"[green]状态：已生效（ACTIVE）[/green] [dim]装配顺序第 {position} / {len(solution.order)} 位[/dim]"
        )
    elif status == "disabled":
        console.print("[yellow]状态：已禁用（disabled）[/yellow] [dim]清单声明 disabled=true，装配时被排除[/dim]")
    else:
        console.print(f"[red]状态：未生效（INACTIVE）[/red] [dim]缺少 {len(missing)} 个必需依赖[/dim]")

    if target.inject:
        # 失活原因已在下文「缺失必需依赖」逐键展开，此处不重复列同一批键
        explained = {"missing", "inactive-provider"} if status == "inactive" else set()
        shown = [item for item in injects if item["state"] not in explained]
        if shown:
            console.print("[bold]依赖[/bold]")
            for item in shown:
                key = item["key"]
                if item["state"] == "ok":
                    console.print(f"  [green]{key}[/green] [dim]← {item['provider']}[/dim]")
                elif item["state"] == "optional-missing":
                    console.print(f"  [yellow]{key}[/yellow] [dim]← 未提供，按可选依赖降级[/dim]")
                else:
                    console.print(f"  [red]{key}[/red] [dim]← 无提供者[/dim]")

    if missing:
        console.print("[bold]缺失必需依赖[/bold]")
        for entry in missing:
            for line in _render_missing(entry):
                console.print(line)

    raise typer.Exit(0 if status == "active" else 1)
