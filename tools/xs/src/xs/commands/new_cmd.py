"""
xs new 命令模块
从模板创建新项目
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ..config import ProjectType, find_workspace_root
from ..discovery import discover_projects, find_project_by_name

console = Console()


def kebab_to_snake(name: str) -> str:
    """
    将 kebab-case 名称转换为 snake_case

    Args:
        name: kebab-case 名称

    Returns:
        snake_case 名称
    """
    return name.replace("-", "_").lower()


def validate_project_name(name: str) -> bool:
    """
    验证项目名称是否合法

    Args:
        name: 项目名称

    Returns:
        是否合法
    """
    if not name:
        return False
    if not re.match(r"^[a-z][a-z0-9-]*$", name):
        return False
    if "--" in name:
        return False
    if name.startswith("-") or name.endswith("-"):
        return False
    return True


def get_templates_dir(workspace_root: Path) -> Path:
    """
    获取模板目录路径

    Args:
        workspace_root: 工作区根目录

    Returns:
        模板目录路径
    """
    return workspace_root / "tools" / "templates"


def copy_template_recursive(
    src: Path,
    dst: Path,
    replacements: dict[str, str],
) -> None:
    """
    递归复制模板目录，并替换文件内容和路径中的占位符

    Args:
        src: 源目录路径
        dst: 目标目录路径
        replacements: 替换字典，key 为占位符，value 为替换值
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name.startswith(".") and item.name not in (".gitignore",):
            continue

        src_path = item
        dst_name = item.name
        for placeholder, value in replacements.items():
            dst_name = dst_name.replace(placeholder, value)
        dst_path = dst / dst_name

        if item.is_dir():
            copy_template_recursive(src_path, dst_path, replacements)
        else:
            if item.suffix in (".pyc", ".pyo", ".so", ".dll", ".exe"):
                continue

            try:
                content = item.read_text(encoding="utf-8")
                for placeholder, value in replacements.items():
                    content = content.replace(placeholder, value)
                dst_path.write_text(content, encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                shutil.copy2(src_path, dst_path)


def create_new_project(
    name: str = typer.Argument(..., help="项目名称（kebab-case）"),
    project_type: ProjectType = typer.Option(..., "--type", "-t", help="项目类型（python/native/static）"),
    is_app: bool = typer.Option(False, "--app", "-a", help="创建为应用（放置在 apps/ 目录）"),
    force: bool = typer.Option(False, "--force", "-f", help="强制创建（覆盖已存在的目录）"),
) -> None:
    """
    从模板创建新项目

    Python 和 Native 项目默认放置在 libs/ 目录，使用 --app 可放置在 apps/。
    Static 项目默认放置在 apps/ 目录。
    """
    if not validate_project_name(name):
        console.print("[red]错误: 项目名称不合法[/red]")
        console.print("[yellow]项目名称应使用小写字母、数字和连字符，以字母开头[/yellow]")
        console.print("[yellow]例如: my-project, xuan-utils[/yellow]")
        raise typer.Exit(1)

    if project_type == "other":
        console.print("[red]错误: 不支持创建 other 类型的项目[/red]")
        raise typer.Exit(1)

    try:
        workspace_root = find_workspace_root()
    except RuntimeError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)

    projects = discover_projects(workspace_root)
    if find_project_by_name(projects, name) is not None:
        console.print(f"[red]错误: 项目 '{name}' 已存在[/red]")
        if not force:
            raise typer.Exit(1)

    templates_dir = get_templates_dir(workspace_root)
    template_dir = templates_dir / project_type

    if not template_dir.exists():
        console.print(f"[red]错误: 模板目录不存在: {template_dir}[/red]")
        raise typer.Exit(1)

    if project_type == "static":
        target_parent = workspace_root / "apps"
    else:
        target_parent = workspace_root / ("apps" if is_app else "libs")

    target_dir = target_parent / name

    if target_dir.exists():
        if not force:
            console.print(f"[red]错误: 目标目录已存在: {target_dir}[/red]")
            console.print("[yellow]使用 --force 选项可强制覆盖[/yellow]")
            raise typer.Exit(1)
        else:
            console.print(f"[yellow]警告: 覆盖已存在的目录: {target_dir}[/yellow]")
            shutil.rmtree(target_dir)

    package_name = kebab_to_snake(name)

    replacements = {
        "{{name}}": name,
        "{{package_name}}": package_name,
    }

    console.print(f"[cyan]正在创建 {project_type} 项目: [bold]{name}[/bold][/cyan]")
    console.print(f"[dim]目标目录: {target_dir.relative_to(workspace_root)}[/dim]")
    console.print(f"[dim]包名: {package_name}[/dim]")
    console.print()

    try:
        copy_template_recursive(template_dir, target_dir, replacements)
    except Exception as e:
        console.print(f"[red]错误: 创建项目失败: {e}[/red]")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        raise typer.Exit(1)

    console.print(
        Panel.fit(
            f"[bold green]✓ 项目 '{name}' 创建成功！[/bold green]\n\n"
            f"[bold]下一步:[/bold]\n"
            f"  1. 进入项目目录: [cyan]cd {target_dir.relative_to(workspace_root)}[/cyan]\n"
            f"  2. 安装依赖: [cyan]pdm install[/cyan]\n"
            f"  3. 开始开发！\n",
            title="项目创建完成",
            border_style="green",
        )
    )
