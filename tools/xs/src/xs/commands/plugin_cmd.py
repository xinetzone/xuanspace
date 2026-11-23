"""``xs plugin`` 命令：清单解析、装配图、求解与运行时装配（P1-1 / P1-2 / P1-3）。

数据来源为「静态声明 ∪ entry points」：前者扫描工作区内各项目 ``pyproject.toml``
的 ``[tool.xuan.plugin]``，后者读取 ``xuan.plugins`` 入口点。``apply --dry-run``
输出求解计划（定序/隔离域/可解性）；不加 ``--dry-run`` 时对内置插件执行真实
装配，并以「声明 / 实际注册」逐键对账收尾。
"""

import json
from importlib.metadata import entry_points
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from xuan_build import PLUGIN as BUILD_PLUGIN
from xuan_ffi import PLUGIN as FFI_PLUGIN
from xuan_plugin.assembly import assemble
from xuan_plugin.manifest import (
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

    if graph_data.unresolved:
        console.print("\n[red]未解析的必需依赖：[/red]")
        for consumer, key in graph_data.unresolved:
            console.print(f"  {consumer} 需要 {key}")
    if graph_data.conflicts:
        console.print("\n[red]能力键冲突（无唯一胜出者：全 single 多提供者，或 multi 优先级并列）：[/red]")
        for key, providers in graph_data.conflicts:
            console.print(f"  {key}：{', '.join(providers)}")


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
