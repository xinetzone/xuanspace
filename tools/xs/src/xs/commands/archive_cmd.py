"""
xs archive 命令模块
将子项目归档到 attic/ 目录
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

ARCHIVE_MARKER = "> ⚠️ **此项目已归档** — 不再活跃维护，仅供参考。\n"

app = typer.Typer(help="项目归档命令", add_completion=False)


def _find_project(name: str) -> Optional[Path]:
    """在 apps/ 和 libs/ 中查找指定项目"""
    from ..config import find_workspace_root

    root = find_workspace_root()
    for base in ("apps", "libs", "tools"):
        for subdir in (root / base).iterdir():
            if subdir.is_dir() and subdir.name == name:
                return subdir
    return None


def _get_project_type(project_path: Path) -> str:
    """检测项目类型"""
    if (project_path / "pyproject.toml").exists():
        return "python"
    if (project_path / "CMakeLists.txt").exists():
        return "native"
    if (project_path / "index.html").exists():
        return "static"
    return "other"


def _find_project_in_attic(name: str) -> Optional[Path]:
    """在 attic/ 中查找项目"""
    from ..config import find_workspace_root

    root = find_workspace_root()
    attic = root / "attic" / name
    if attic.exists():
        return attic
    return None


@app.command()
def archive(
    name: str = typer.Argument(..., help="要归档的项目名称"),
    reason: Optional[str] = typer.Option(None, "--reason", "-r", help="归档原因"),
    force: bool = typer.Option(False, "--force", "-f", help="强制归档，跳过确认"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览归档操作，不实际执行"),
) -> None:
    """将子项目从 apps/ 或 libs/ 移动到 attic/"""
    project_path = _find_project(name)
    if not project_path:
        console.print(f"[red]错误: 未找到项目 '{name}'[/red]")
        console.print("[dim]使用 'xs list' 查看所有项目[/dim]")
        raise typer.Exit(1)

    from ..config import find_workspace_root

    root = find_workspace_root()
    attic_path = root / "attic" / name
    proj_type = _get_project_type(project_path)

    console.print()
    console.print(Panel.fit(
        f"[bold yellow]归档项目: {name}[/bold yellow]",
        border_style="yellow",
    ))
    console.print(f"  源路径: {project_path.relative_to(root)}")
    console.print(f"  目标路径: {attic_path.relative_to(root)}")
    console.print(f"  项目类型: {proj_type}")
    if reason:
        console.print(f"  归档原因: {reason}")
    console.print()

    if dry_run:
        console.print("[bold cyan][DRY RUN] 以下操作将被执行:[/bold cyan]")
        console.print(f"  1. 移动 {project_path.relative_to(root)} -> {attic_path.relative_to(root)}")
        console.print("  2. 在 README.md 中添加归档标记")
        console.print("  3. 提示更新根目录 README 项目索引")
        return

    if attic_path.exists():
        console.print(f"[red]错误: attic/ 中已存在 '{name}'[/red]")
        raise typer.Exit(1)

    if not force:
        typer.confirm(f"确认归档 '{name}'?", abort=True)

    # 移动项目
    project_path.rename(attic_path)
    console.print(f"[green]✓ 已移动: {project_path.relative_to(root)} -> {attic_path.relative_to(root)}[/green]")

    # 添加归档标记到 README
    readme = attic_path / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        if not content.startswith(ARCHIVE_MARKER):
            readme.write_text(ARCHIVE_MARKER + content, encoding="utf-8")
            console.print("[green]✓ 已在 README.md 中添加归档标记[/green]")

    # 更新根 README 项目索引提示
    console.print()
    console.print("[bold yellow]后续手动步骤:[/bold yellow]")
    console.print("  1. 更新根目录 README.md 中的项目索引表格")
    console.print("  2. 提交变更: git add attic/ && git commit -m 'chore: archive {name}'")
    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ 项目 '{name}' 已归档到 attic/[/bold green]\n\n"
        f"[dim]如需恢复，请将 attic/{name}/ 移回原位置[/dim]",
        border_style="green",
    ))


@app.command()
def unarchive(
    name: str = typer.Argument(..., help="要恢复的项目名称"),
    target_dir: Optional[str] = typer.Option(None, "--target", "-t", help="目标目录 (apps 或 libs)，默认为自动检测"),
    force: bool = typer.Option(False, "--force", "-f", help="强制恢复，跳过确认"),
    dry_run: bool = typer.Option(False, "--dry-run", help="预览恢复操作，不实际执行"),
) -> None:
    """将归档项目从 attic/ 恢复到 apps/ 或 libs/"""
    attic_project = _find_project_in_attic(name)
    if not attic_project:
        console.print(f"[red]错误: attic/ 中未找到 '{name}'[/red]")
        raise typer.Exit(1)

    from ..config import find_workspace_root

    root = find_workspace_root()
    proj_type = _get_project_type(attic_project)

    if target_dir:
        if target_dir not in ("apps", "libs", "tools"):
            console.print(f"[red]错误: 目标目录必须是 apps、libs 或 tools，收到: '{target_dir}'[/red]")
            raise typer.Exit(1)
        dest = root / target_dir / name
    else:
        if proj_type in ("python", "native"):
            dest = root / "libs" / name
        else:
            dest = root / "apps" / name

    console.print()
    console.print(f"[bold]恢复项目: {name}[/bold]")
    console.print(f"  源路径: {attic_project.relative_to(root)}")
    console.print(f"  目标路径: {dest.relative_to(root)}")
    console.print()

    if dry_run:
        return

    if dest.exists():
        console.print(f"[red]错误: 目标路径 '{dest.relative_to(root)}' 已存在[/red]")
        raise typer.Exit(1)

    if not force:
        typer.confirm(f"确认恢复 '{name}'?", abort=True)

    attic_project.rename(dest)
    console.print(f"[green]✓ 已恢复: {attic_project.relative_to(root)} -> {dest.relative_to(root)}[/green]")

    # 移除归档标记
    readme = dest / "README.md"
    if readme.exists():
        content = readme.read_text(encoding="utf-8")
        if content.startswith(ARCHIVE_MARKER):
            readme.write_text(content[len(ARCHIVE_MARKER):], encoding="utf-8")
            console.print("[green]✓ 已移除 README.md 中的归档标记[/green]")

    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ 项目 '{name}' 已恢复[/bold green]",
        border_style="green",
    ))