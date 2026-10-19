"""
xs build 命令模块
构建工作区中的项目
"""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import ProjectType, find_workspace_root, load_pyproject
from ..discovery import ProjectInfo, discover_projects, find_project_by_name

console = Console()


def _auto_build_enabled(project: ProjectInfo) -> bool:
    """批量构建（--type all / --type <type>）时是否包含该项目

    项目可在其 pyproject.toml 中声明 [tool.xs] auto-build = false，
    用于标记依赖系统级 C++ 库（如 Protobuf/OpenBLAS）而无法在裸环境
    批量构建的项目（如需 Docker 黄金环境构建的 caffe-ffi）。
    """
    if project.pyproject_path is None:
        return True
    try:
        data = load_pyproject(project.pyproject_path)
    except Exception:
        return True
    return bool(data.get("tool", {}).get("xs", {}).get("auto-build", True))


def _build_python_project(project_path: Path) -> bool:
    """
    构建纯 Python 项目

    Args:
        project_path: 项目目录路径

    Returns:
        构建是否成功
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel"],
            cwd=project_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        console.print("[red]构建失败:[/red]")
        if e.stdout:
            console.print(f"[red]{e.stdout}[/red]")
        if e.stderr:
            console.print(f"[red]{e.stderr}[/red]")
        return False


def _build_native_project(project_path: Path) -> bool:
    """
    构建 C/C++ 原生扩展项目

    Args:
        project_path: 项目目录路径

    Returns:
        构建是否成功
    """
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            cwd=project_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as first_err:
        console.print("[yellow]pip install -e 失败，回退 python -m build --wheel[/yellow]")
        try:
            subprocess.run(
                [sys.executable, "-m", "build", "--wheel"],
                cwd=project_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            # ninja/CMake 的 FAILED: 详情写入 stdout，两路输出都必须保留
            console.print("[red]构建失败:[/red]")
            for err in (first_err, e):
                if err.stdout:
                    console.print(f"[red]{err.stdout}[/red]")
                if err.stderr:
                    console.print(f"[red]{err.stderr}[/red]")
            return False


def _get_project_for_build(
    workspace_root: Path,
    project_name: str | None,
) -> Path | None:
    """
    获取要构建的项目路径

    Args:
        workspace_root: 工作区根目录
        project_name: 指定的项目名称（可选）

    Returns:
        项目路径，如果未找到返回 None
    """
    if project_name:
        projects = discover_projects(workspace_root)
        project = find_project_by_name(projects, project_name)
        if project is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            return None
        return project.path

    cwd = Path.cwd().resolve()
    try:
        cwd.relative_to(workspace_root.resolve())
    except ValueError:
        console.print("[red]错误: 当前目录不在工作区内[/red]")
        return None

    projects = discover_projects(workspace_root)
    for project in projects:
        try:
            cwd.relative_to(project.path.resolve())
            return project.path
        except ValueError:
            continue

    console.print("[yellow]当前目录不属于任何项目，请使用 --project 指定项目名称[/yellow]")
    return None


def _build_single_project(project_path: Path, project_type: ProjectType) -> bool:
    """
    构建单个项目

    Args:
        project_path: 项目路径
        project_type: 项目类型

    Returns:
        构建是否成功
    """
    project_name = project_path.name

    if project_type == "static":
        console.print(f"[yellow]跳过静态项目: {project_name} (无需构建)[/yellow]")
        return True

    if project_type == "other":
        console.print(f"[yellow]跳过未知类型项目: {project_name}[/yellow]")
        return True

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"正在构建 {project_name}...", total=None)

        if project_type == "python":
            success = _build_python_project(project_path)
        elif project_type == "native":
            success = _build_native_project(project_path)
        else:
            success = False

        progress.update(task, completed=True)

    if success:
        console.print(f"[green]✓ {project_name} 构建成功[/green]")
    else:
        console.print(f"[red]✗ {project_name} 构建失败[/red]")

    return success


def build_project(
    project_name: str | None = typer.Option(None, "--project", "-p", help="要构建的项目名称"),
    build_type: str | None = typer.Option(None, "--type", "-t", help="按类型构建（python/native/all）"),
) -> None:
    """
    构建项目

    支持构建指定项目、当前目录项目，或按类型批量构建。
    Python 项目使用 python -m build --wheel，
    Native 项目使用 pip install -e . 或 python -m build --wheel。
    """
    try:
        workspace_root = find_workspace_root()
    except RuntimeError as e:
        console.print(f"[red]错误: {e}[/red]")
        raise typer.Exit(1)

    if build_type and build_type not in ("python", "native", "all"):
        console.print(f"[red]错误: 不支持的构建类型 '{build_type}'，请使用 python/native/all[/red]")
        raise typer.Exit(1)

    projects = discover_projects(workspace_root)

    if build_type == "all":
        selected = list(projects)
    elif build_type:
        selected = [p for p in projects if p.project_type == build_type]
    else:
        selected = []

    if build_type:
        # 批量构建时尊重项目级 auto-build=false 退出声明
        projects_to_build = []
        for p in selected:
            if _auto_build_enabled(p):
                projects_to_build.append(p)
            else:
                console.print(f"[yellow]跳过 {p.name}（auto-build=false，需专用环境构建）[/yellow]")
    elif project_name:
        project = find_project_by_name(projects, project_name)
        if project is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        projects_to_build = [project]
    else:
        project_path = _get_project_for_build(workspace_root, None)
        if project_path is None:
            raise typer.Exit(1)
        current_project = None
        for p in projects:
            if p.path.resolve() == project_path.resolve():
                current_project = p
                break
        if current_project is None:
            console.print("[red]错误: 无法确定当前项目[/red]")
            raise typer.Exit(1)
        projects_to_build = [current_project]

    if not projects_to_build:
        console.print("[yellow]没有找到符合条件的项目[/yellow]")
        return

    console.print(f"[cyan]将构建 {len(projects_to_build)} 个项目[/cyan]")
    console.print()

    success_count = 0
    fail_count = 0

    for project in projects_to_build:
        if _build_single_project(project.path, project.project_type):
            success_count += 1
        else:
            fail_count += 1
        console.print()

    console.print("[bold]构建完成:[/bold]")
    console.print(f"  [green]成功: {success_count}[/green]")
    if fail_count > 0:
        console.print(f"  [red]失败: {fail_count}[/red]")
        raise typer.Exit(1)
