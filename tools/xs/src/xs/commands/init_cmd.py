"""
xs init 命令模块
初始化新项目或工作区
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ..config import check_python_version, find_workspace_root, get_python_version

console = Console()


def _check_python() -> bool:
    version = get_python_version()
    if check_python_version((3, 13)):
        console.print(f"[green]✓ Python {version}[/green]")
        return True
    else:
        console.print(f"[red]✗ Python {version} < 3.13，请升级 Python[/red]")
        console.print("[dim]  下载地址: https://www.python.org/downloads/[/dim]")
        return False


def _check_git() -> bool:
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            console.print(f"[green]✓ {result.stdout.strip()}[/green]")
            return True
    except FileNotFoundError:
        pass
    console.print("[yellow]⚠ Git 未安装（可选但推荐）[/yellow]")
    return False


def init_workspace() -> None:
    """初始化当前目录为 Xuanspace 工作区（验证环境）"""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Xuanspace（玄境）工作区初始化[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    cwd = Path.cwd()
    pyproject = cwd / "pyproject.toml"

    if pyproject.exists():
        try:
            root = find_workspace_root(cwd)
            if root == cwd.resolve():
                console.print(f"[green]✓ 当前目录已是 Xuanspace 工作区[/green]")
                console.print(f"[dim]工作区根目录: {root}[/dim]")
            else:
                console.print(f"[yellow]⚠ 上级目录 {root} 是工作区根目录[/yellow]")
        except RuntimeError:
            console.print("[yellow]⚠ 当前目录有 pyproject.toml 但不是 Xuanspace 工作区[/yellow]")
    else:
        console.print("[yellow]当前目录不是 Xuanspace 工作区[/yellow]")
        console.print("[dim]请在 Xuanspace 仓库根目录运行此命令[/dim]")

    console.print()
    console.print("[bold]环境检查:[/bold]")
    py_ok = _check_python()
    _check_git()
    console.print()

    if not py_ok:
        raise typer.Exit(1)

    console.print(Panel.fit(
        "[bold green]✓ 环境就绪！[/bold green]\n\n"
        "[bold]快速开始:[/bold]\n"
        "  xs list         查看所有项目\n"
        "  xs doctor       环境诊断\n"
        "  xs new --type python my-lib  创建新项目\n"
        "  xs build        构建项目\n"
        "  xs docs build   构建文档",
        border_style="green",
    ))
