"""
xs deps 命令模块
依赖检查和管理
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import load_pyproject
from ..discovery import build_dependency_graph, discover_projects, find_project_by_name

console = Console()


def _parse_dep_spec(spec: str) -> tuple[str, str]:
    """解析依赖规范为 (name, constraint)"""
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", spec)
    if match:
        return match.group(1), match.group(2)
    return spec, ""


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def deps_check(
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="检查指定项目"),
) -> None:
    """检查依赖一致性，检测版本冲突"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    projects = discover_projects(workspace_root)
    dep_graph = build_dependency_graph(projects)

    if project_name:
        proj = find_project_by_name(projects, project_name)
        if proj is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        projects = [proj]

    all_requirements: dict[str, list[tuple[str, str]]] = {}

    for project in projects:
        if not project.pyproject_path:
            continue
        try:
            data = load_pyproject(project.pyproject_path)
            deps = data.get("project", {}).get("dependencies", [])
            for dep in deps:
                name, constraint = _parse_dep_spec(dep)
                norm_name = _normalize_name(name)
                if norm_name not in all_requirements:
                    all_requirements[norm_name] = []
                all_requirements[norm_name].append((project.name, constraint))
        except Exception:
            continue

    table = Table(title="依赖版本检查", show_header=True, header_style="bold magenta")
    table.add_column("包名", style="cyan")
    table.add_column("版本约束", style="green")
    table.add_column("引用项目")

    conflicts = 0
    for pkg, refs in sorted(all_requirements.items()):
        constraints = list({c for _, c in refs if c})
        projects_list = ", ".join(p for p, _ in refs)
        if len(constraints) > 1:
            conflicts += 1
            table.add_row(
                f"[yellow]{pkg}[/yellow]",
                " / ".join(constraints),
                f"[red]{projects_list}[/red] ⚠ 冲突",
            )
        else:
            constraint = constraints[0] if constraints else "any"
            table.add_row(pkg, constraint, projects_list)

    console.print(table)

    internal_deps_count = sum(len(deps) for deps in dep_graph.values())
    console.print()
    console.print(f"内部依赖引用: {internal_deps_count}")
    if conflicts > 0:
        console.print(f"[yellow]⚠ 发现 {conflicts} 个潜在版本冲突[/yellow]")
    else:
        console.print("[green]✓ 所有第三方依赖版本一致[/green]")


def deps_tree(
    project_name: Optional[str] = typer.Option(None, "--project", "-p", help="显示指定项目的依赖树"),
    depth: int = typer.Option(3, "--depth", "-d", help="依赖树最大深度"),
) -> None:
    """显示项目依赖树"""
    from ..config import find_workspace_root

    workspace_root = find_workspace_root()
    projects = discover_projects(workspace_root)
    dep_graph = build_dependency_graph(projects)
    project_names = {p.name: p for p in projects}

    roots = [p for p in projects if not any(p.name in deps for deps in dep_graph.values())]
    if project_name:
        proj = find_project_by_name(projects, project_name)
        if proj is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        roots = [proj]

    def _print_tree(name: str, prefix: str = "", current_depth: int = 0, visited: set | None = None):
        if visited is None:
            visited = set()
        if current_depth >= depth:
            console.print(f"{prefix}[dim]...[/dim]")
            return
        if name in visited:
            console.print(f"{prefix}[yellow]{name}[/yellow] [dim](circular)[/dim]")
            return
        visited = visited | {name}

        children = dep_graph.get(name, [])
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "└── " if is_last else "├── "
            child_proj = project_names.get(child)
            color = "green" if child_proj and child_proj.project_type == "python" else "cyan"
            console.print(f"{prefix}{connector}[{color}]{child}[/{color}]")
            extension = "    " if is_last else "│   "
            _print_tree(child, prefix + extension, current_depth + 1, visited)

    for root in roots:
        console.print(f"[bold cyan]{root.name}[/bold cyan] [dim]({root.project_type})[/dim]")
        _print_tree(root.name)
        console.print()


app = typer.Typer(help="依赖管理命令", add_completion=False)
app.command("check")(deps_check)
app.command("tree")(deps_tree)
