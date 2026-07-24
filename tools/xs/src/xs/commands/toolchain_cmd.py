"""
xs toolchain 命令模块
工具链管理：检查和安装开发工具
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console

from .doctor_cmd import run_doctor

console = Console()

app = typer.Typer(help="工具链管理命令", add_completion=False)


INSTALLABLE_TOOLS = {
    "cmake": {
        "pip_package": "cmake",
        "description": "CMake 构建系统",
    },
    "ninja": {
        "pip_package": "ninja",
        "description": "Ninja 构建工具",
    },
    "build": {
        "pip_package": "build",
        "description": "Python 构建工具",
    },
}


def _check_command() -> None:
    """toolchain check 子命令：检查工具链，缺失必需工具时返回非零退出码"""
    exit_code = run_doctor(check_mode=True)
    raise typer.Exit(exit_code)


def _list_tools() -> None:
    """toolchain list 子命令：列出可安装的工具"""
    from rich.table import Table

    table = Table(
        title="[bold]可安装工具列表[/bold]",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("工具名", style="cyan", no_wrap=True)
    table.add_column("pip 包名", style="green")
    table.add_column("描述")

    for tool_name, tool_info in INSTALLABLE_TOOLS.items():
        table.add_row(
            tool_name,
            tool_info["pip_package"],
            tool_info["description"],
        )

    console.print(table)
    console.print()
    console.print("[dim]使用 xs toolchain install <tool> 安装工具[/dim]")


def _install_tool(
    tool_name: str = typer.Argument(..., help="要安装的工具名称"),
    user: bool = typer.Option(True, "--user/--system", help="安装到用户目录或系统目录"),
    upgrade: bool = typer.Option(False, "--upgrade", "-U", help="升级已安装的工具"),
) -> None:
    """toolchain install 子命令：安装指定工具"""
    if tool_name not in INSTALLABLE_TOOLS:
        console.print(f"[red]错误: 未知的工具 '{tool_name}'[/red]")
        console.print(f"[yellow]可安装的工具: {', '.join(INSTALLABLE_TOOLS.keys())}[/yellow]")
        raise typer.Exit(1)

    tool_info = INSTALLABLE_TOOLS[tool_name]
    pip_package = tool_info["pip_package"]

    console.print(f"[cyan]正在安装 {tool_info['description']} ({pip_package})...[/cyan]")

    cmd = [sys.executable, "-m", "pip", "install"]
    if user:
        cmd.append("--user")
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(pip_package)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        console.print(f"[green]✓ {tool_name} 安装成功！[/green]")
        if result.stdout:
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ {tool_name} 安装失败[/red]")
        if e.stderr:
            console.print(f"[red]{e.stderr.strip()}[/red]")
        raise typer.Exit(1)


app.command("check")(_check_command)
app.command("list")(_list_tools)
app.command("install")(_install_tool)
