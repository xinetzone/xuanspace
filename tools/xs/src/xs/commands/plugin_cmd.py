"""``xs plugin`` 命令：清单解析与装配图展示（P1-1）。

数据来源为「静态声明 ∪ entry points」：前者扫描工作区内各项目 ``pyproject.toml``
的 ``[tool.xuan.plugin]``，后者读取 ``xuan.plugins`` 入口点。本命令只读取与展示，
不激活插件；求解与回滚由 P1-2 的装配器负责。
"""

import json
from importlib.metadata import entry_points
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from xuan_plugin.manifest import (
    PluginManifest,
    build_graph,
    manifest_from_pyproject,
    merge_manifests,
)

from ..config import find_workspace_root, load_pyproject
from ..discovery import discover_projects

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

    table = Table(title="插件清单", show_lines=False)
    table.add_column("ID", style="bold cyan")
    table.add_column("VERSION", style="green")
    table.add_column("LIFECYCLE")
    table.add_column("PRIORITY", justify="right")
    table.add_column("PROVIDES", justify="right")
    table.add_column("INJECTS", justify="right")
    table.add_column("SOURCE", style="dim")
    for manifest in manifests:
        table.add_row(
            manifest.id,
            manifest.version,
            manifest.lifecycle,
            str(manifest.priority),
            str(len(manifest.provide)),
            str(len(manifest.inject)),
            manifest.source or "-",
        )
    console.print(table)
    keys = {p.key for m in manifests for p in m.provide}
    console.print(f"[dim]共 {len(manifests)} 个插件，{len(keys)} 个能力键[/dim]")


@app.command("graph")
def graph(
    dot: bool = typer.Option(False, "--dot", help="输出 Graphviz DOT"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出"),
) -> None:
    """输出候选节点与依赖边，并标注未解析依赖与冲突键。"""
    try:
        manifests = collect_manifests(find_workspace_root())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]清单解析失败：{exc}[/red]")
        raise typer.Exit(1) from exc

    graph_data = build_graph(manifests)

    if dot:
        typer.echo(graph_data.to_dot())
        return
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "nodes": list(graph_data.nodes),
                    "edges": [{"consumer": e.consumer, "key": e.key, "provider": e.provider} for e in graph_data.edges],
                    "unresolved": [{"consumer": c, "key": k} for c, k in graph_data.unresolved],
                    "conflicts": [{"key": k, "providers": list(p)} for k, p in graph_data.conflicts],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    key_count = len({p.key for m in manifests for p in m.provide})
    console.print(
        f"[dim]插件 {len(graph_data.nodes)} 个，能力键 {key_count} 个，"
        f"依赖边 {len(graph_data.edges)} 条，未解析 {len(graph_data.unresolved)}，"
        f"冲突 {len(graph_data.conflicts)}[/dim]"
    )

    by_id = {m.id: m for m in manifests}
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
                style = "yellow" if edge.key not in {k for _, k in graph_data.unresolved} else "red"
                console.print(f"  [{style}]inject[/{style}] {edge.key} [dim]→ 未解析[/dim]")
            else:
                console.print(f"  inject {edge.key} [dim]→ {edge.provider}[/dim]")

    if graph_data.unresolved:
        console.print("\n[red]未解析的必需依赖：[/red]")
        for consumer, key in graph_data.unresolved:
            console.print(f"  {consumer} 需要 {key}")
    if graph_data.conflicts:
        console.print("\n[red]能力键冲突（同键多提供者且未声明 multi）：[/red]")
        for key, providers in graph_data.conflicts:
            console.print(f"  {key}：{', '.join(providers)}")
