"""
xs list 命令模块
列出工作区中的所有项目
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..config import ProjectType, find_workspace_root
from ..discovery import discover_projects, filter_projects

console = Console()


def _get_type_color(project_type: ProjectType) -> str:
    """获取项目类型对应的颜色"""
    colors = {
        ProjectType.PYTHON: "green",
        ProjectType.NATIVE: "cyan",
        ProjectType.STATIC: "yellow",
        ProjectType.OTHER: "white",
    }
    return colors.get(project_type, "white")


def _get_type_display(project_type: ProjectType) -> str:
    """获取项目类型的显示名称"""
    displays = {
        ProjectType.PYTHON: "Python",
        ProjectType.NATIVE: "Native",
        ProjectType.STATIC: "Static",
        ProjectType.OTHER: "Other",
    }
    return displays.get(project_type, str(project_type))


def list_projects(
    project_type: Optional[ProjectType] = typer.Option(
        None, "--type", "-t", help="按项目类型过滤"
    ),
    directory: Optional[str] = typer.Option(
        None, "--dir", "-d", help="按目录过滤（apps/libs）"
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="以 JSON 格式输出"
    ),
) -> None:
    """
    列出工作区中的所有项目

    显示项目名称、类型、版本和路径信息。
    支持按类型和目录过滤，支持 JSON 格式输出。
    """
    try:
        workspace_root = find_workspace_root()
        projects = discover_projects(workspace_root)
        projects = filter_projects(projects, project_type, directory)
        projects.sort(key=lambda p: p.name)

        if json_output:
            output = []
            for p in projects:
                output.append({
                    "name": p.name,
                    "type": p.project_type,
                    "version": p.version,
                    "path": str(p.path.relative_to(workspace_root)),
                    "dependencies": p.dependencies,
                })
            console.print(json.dumps(output, indent=2, ensure_ascii=False))
            return

        if not projects:
            console.print("[yellow]未找到任何项目[/yellow]")
            return

        table = Table(
            title=f"[bold]Xuanspace 项目列表[/bold] (工作区: {workspace_root.name})",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("NAME", style="cyan", no_wrap=True)
        table.add_column("TYPE", style="bold")
        table.add_column("VERSION", style="green")
        table.add_column("PATH", style="dim")

        type_counts: dict[str, int] = {}
        for p in projects:
            type_counts[p.project_type] = type_counts.get(p.project_type, 0) + 1
            rel_path = p.path.relative_to(workspace_root)
            table.add_row(
                p.name,
                f"[{_get_type_color(p.project_type)}]{_get_type_display(p.project_type)}[/{_get_type_color(p.project_type)}]",
                p.version,
                str(rel_path),
            )

        console.print(table)

        summary_parts = [f"共 [bold]{len(projects)}[/bold] 个项目"]
        for ptype, count in type_counts.items():
            summary_parts.append(f"{_get_type_display(ptype)}: {count}")
        console.print("  ".join(summary_parts))

    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)
