"""``xs plugin`` 命令：清单解析、装配图与求解计划（P1-1 / P1-2）。

数据来源为「静态声明 ∪ entry points」：前者扫描工作区内各项目 ``pyproject.toml``
的 ``[tool.xuan.plugin]``，后者读取 ``xuan.plugins`` 入口点。本命令只读取与展示，
不激活插件；``apply --dry-run`` 输出求解计划（定序/隔离域/可解性），真实装配
待 P1-3 提供运行时插件实现后接线。
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
from xuan_plugin.solver import SolveError, solve

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


@app.command("apply")
def apply_plugins(
    dry_run: bool = typer.Option(False, "--dry-run", help="只输出求解计划，不执行装配"),
    json_output: bool = typer.Option(False, "--json", "-j", help="以 JSON 格式输出"),
) -> None:
    """求解装配计划：定序、隔离域、可解性诊断（``--dry-run``）。

    真实装配需要各插件提供 ``Plugin`` 实现，P1-3 接线前只交付求解面；
    未加 ``--dry-run`` 时显式拒绝，避免把「什么都没做」伪装成装配成功。
    """
    if not dry_run:
        console.print("[red]真实装配尚未接线（P1-3 提供运行时插件实现），请使用 --dry-run。[/red]")
        raise typer.Exit(1)

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
