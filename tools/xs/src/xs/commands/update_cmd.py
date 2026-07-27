"""
xs update 命令模块
更新子模块和依赖
"""


import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import find_workspace_root

console = Console()


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except FileNotFoundError:
        return False, "Git 未安装"


def update_submodules(recursive: bool = True) -> bool:
    workspace_root = find_workspace_root()
    cmd = ["submodule", "update", "--remote"]
    if recursive:
        cmd.append("--recursive")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("正在更新 git 子模块...", total=None)
        success, output = _run_git(cmd, cwd=workspace_root)
        progress.update(task, completed=True)
    if success:
        console.print("[green]✓ 子模块更新成功[/green]")
        if output:
            console.print(f"[dim]{output}[/dim]")
    else:
        console.print(f"[red]✗ 子模块更新失败: {output}[/red]")
    return success


def install_dependencies() -> bool:
    workspace_root = find_workspace_root()
    pyproject = workspace_root / "pyproject.toml"
    if not pyproject.exists():
        console.print("[yellow]未找到根 pyproject.toml，跳过依赖安装[/yellow]")
        return True

    pdm_lock = workspace_root / "pdm.lock"
    uv_lock = workspace_root / "uv.lock"

    if pdm_lock.exists() or (workspace_root / ".pdm-python").exists():
        console.print("[cyan]检测到 PDM 环境，使用 pdm install 更新依赖...[/cyan]")
        cmd = [sys.executable, "-m", "pdm", "install"]
    elif uv_lock.exists():
        console.print("[cyan]检测到 uv 环境，使用 uv sync 更新依赖...[/cyan]")
        cmd = [sys.executable, "-m", "uv", "sync"]
    else:
        console.print("[cyan]使用 pip 安装依赖...[/cyan]")
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("正在安装/更新依赖...", total=None)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=workspace_root)
            progress.update(task, completed=True)
        except FileNotFoundError as e:
            progress.update(task, completed=True)
            console.print(f"[yellow]命令不可用: {e}，跳过依赖更新[/yellow]")
            return True

    if result.returncode == 0:
        console.print("[green]✓ 依赖更新成功[/green]")
        return True
    else:
        console.print("[red]✗ 依赖更新失败[/red]")
        if result.stderr:
            console.print(f"[dim]{result.stderr[:500]}[/dim]")
        return False


def update_cmd(
    skip_submodules: bool = typer.Option(False, "--no-submodules", help="跳过子模块更新"),
    skip_deps: bool = typer.Option(False, "--no-deps", help="跳过依赖安装"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="递归更新子模块"),
) -> None:
    """更新工作区（git 子模块和依赖）"""
    try:
        find_workspace_root()
    except RuntimeError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)

    ok = True
    if not skip_submodules:
        ok = update_submodules(recursive) and ok
    if not skip_deps:
        ok = install_dependencies() and ok

    if not ok:
        raise typer.Exit(1)
    console.print("[bold green]✓ 更新完成[/bold green]")
