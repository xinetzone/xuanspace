"""
xs deps 命令模块
依赖检查、更新和管理
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import typer
from packaging.version import Version
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..config import find_workspace_root, load_pyproject
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
    project_name: str | None = typer.Option(None, "--project", "-p", help="检查指定项目"),
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
    project_name: str | None = typer.Option(None, "--project", "-p", help="显示指定项目的依赖树"),
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
        console.print(f"[bold cyan]{root.name}[/bold cyan] [dim]({root.project_type.value})[/dim]")
        _print_tree(root.name)
        console.print()


def _fetch_latest_version(package: str, cache: dict[str, str | None]) -> str | None:
    if package in cache:
        return cache[package]
    try:
        url = f"https://pypi.org/pypi/{package}/json"
        req = urllib.request.Request(url, headers={"User-Agent": "xuanspace-cli/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            ver = data.get("info", {}).get("version")
            cache[package] = ver
            return ver
    except Exception:
        cache[package] = None
        return None


def _extract_min_version(constraint: str) -> Version | None:
    if not constraint:
        return None
    match = re.search(r">=\s*(\d+(?:\.\d+)*)", constraint)
    if match:
        try:
            return Version(match.group(1))
        except Exception:
            return None
    match = re.search(r"(?:==|~=)\s*(\d+(?:\.\d+)*)", constraint)
    if match:
        try:
            return Version(match.group(1))
        except Exception:
            return None
    return None


def _update_dep_in_content(content: str, dep_name: str, new_version: str) -> tuple[str, bool]:
    pattern = r'("' + re.escape(dep_name) + r'(?:\[[^\]]*\])?)\s*(>=|==|~=|>)\s*\d+(?:\.\d+)*(?:\.[^"\s]*)?\s*"'
    replacement = rf'\1>={new_version}"'
    new_content, count = re.subn(pattern, replacement, content)
    if count > 0:
        return new_content, True
    pattern2 = r'("' + re.escape(dep_name) + r'(?:\[[^\]]*\])?)\s*"'
    replacement2 = rf'\1>={new_version}"'
    new_content, count2 = re.subn(pattern2, replacement2, new_content if count > 0 else content)
    return new_content, (count > 0 or count2 > 0)


def _get_project_deps(pyproject_path: Path) -> tuple[list[str], dict]:
    data = load_pyproject(pyproject_path)
    deps = list(data.get("project", {}).get("dependencies", []))
    opt_deps = data.get("project", {}).get("optional-dependencies", {})
    return deps, opt_deps


def _collect_all_dep_files(projects: list) -> list[tuple[str, Path]]:
    result = []
    for proj in projects:
        if proj.pyproject_path:
            result.append((proj.name, proj.pyproject_path))
    return result


def deps_outdated(
    project_name: str | None = typer.Option(None, "--project", "-p", help="检查指定项目"),
) -> None:
    """检查过期的依赖包"""
    workspace_root = find_workspace_root()
    projects = discover_projects(workspace_root)

    if project_name:
        proj = find_project_by_name(projects, project_name)
        if proj is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        projects = [proj]

    dep_files = _collect_all_dep_files(projects)
    all_deps: dict[str, tuple[str, str, str]] = {}

    for proj_name, pyproject_path in dep_files:
        try:
            deps, opt_deps = _get_project_deps(pyproject_path)
            for dep in deps:
                name, constraint = _parse_dep_spec(dep)
                norm = _normalize_name(name)
                if norm not in all_deps:
                    all_deps[norm] = (name, constraint, proj_name)
            for group_name, group_deps in opt_deps.items():
                for dep in group_deps:
                    name, constraint = _parse_dep_spec(dep)
                    norm = _normalize_name(name)
                    if norm not in all_deps:
                        all_deps[norm] = (name, constraint, f"{proj_name}[{group_name}]")
        except Exception:
            continue

    cache: dict[str, str | None] = {}
    outdated: list[tuple[str, str, str, str]] = []
    up_to_date = 0
    unknown = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"正在检查 {len(all_deps)} 个依赖包...", total=len(all_deps))
        for norm_name, (orig_name, constraint, proj_ref) in sorted(all_deps.items()):
            latest = _fetch_latest_version(norm_name, cache)
            progress.advance(task)
            if latest is None:
                unknown += 1
                continue
            current = _extract_min_version(constraint)
            try:
                latest_ver = Version(latest)
            except Exception:
                unknown += 1
                continue
            if current is None or current < latest_ver:
                outdated.append((orig_name, constraint or "any", latest, proj_ref))
            else:
                up_to_date += 1

    if outdated:
        table = Table(title="过期依赖", show_header=True, header_style="bold magenta")
        table.add_column("包名", style="cyan")
        table.add_column("当前约束", style="yellow", width=20)
        table.add_column("最新版本", style="green", width=12)
        table.add_column("引用位置")
        for name, cur, latest, ref in sorted(outdated, key=lambda x: x[0]):
            table.add_row(name, cur, latest, ref)
        console.print(table)

    console.print()
    console.print(f"[green]✓ {up_to_date} 个已是最新[/green], ", end="")
    if outdated:
        console.print(f"[yellow]⚠ {len(outdated)} 个可更新[/yellow], ", end="")
    if unknown:
        console.print(f"[dim]{unknown} 个无法检查[/dim]", end="")
    console.print()


def deps_update(
    packages: list[str] = typer.Argument(None, help="要更新的包名（空格分隔），不传则更新所有可更新包"),
    project_name: str | None = typer.Option(None, "--project", "-p", help="更新指定项目的依赖"),
    dry_run: bool = typer.Option(False, "--dry-run", help="仅显示将要做的更改，不实际修改文件"),
    no_install: bool = typer.Option(False, "--no-install", help="仅更新 pyproject.toml，不安装依赖"),
) -> None:
    """更新依赖包到最新版本"""
    workspace_root = find_workspace_root()
    projects = discover_projects(workspace_root)

    if project_name:
        proj = find_project_by_name(projects, project_name)
        if proj is None:
            console.print(f"[red]错误: 未找到项目 '{project_name}'[/red]")
            raise typer.Exit(1)
        projects = [proj]

    dep_files = _collect_all_dep_files(projects)
    target_packages = {_normalize_name(p) for p in packages} if packages else None

    cache: dict[str, str | None] = {}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        all_deps_set: set[str] = set()
        dep_file_deps: list[tuple[Path, list[str]]] = []

        for _proj_name, pyproject_path in dep_files:
            try:
                deps, opt_deps = _get_project_deps(pyproject_path)
                all_deps_list = list(deps)
                for gd in opt_deps.values():
                    all_deps_list.extend(gd)
                dep_file_deps.append((pyproject_path, all_deps_list))
                for dep in all_deps_list:
                    name, _ = _parse_dep_spec(dep)
                    all_deps_set.add(_normalize_name(name))
            except Exception:
                dep_file_deps.append((pyproject_path, []))

        if target_packages:
            all_deps_set = {p for p in all_deps_set if p in target_packages}

        task = progress.add_task("正在查询 PyPI 获取最新版本...", total=len(all_deps_set))
        pkg_latest: dict[str, str] = {}
        for norm_name in sorted(all_deps_set):
            latest = _fetch_latest_version(norm_name, cache)
            if latest:
                pkg_latest[norm_name] = latest
            progress.advance(task)

    files_to_update: dict[Path, list[tuple[str, str, str]]] = {}
    for pyproject_path, deps_list in dep_file_deps:
        try:
            content = pyproject_path.read_text(encoding="utf-8")
        except Exception:
            continue
        file_updates = []
        for dep in deps_list:
            orig_name, constraint = _parse_dep_spec(dep)
            norm_name = _normalize_name(orig_name)
            if target_packages and norm_name not in target_packages:
                continue
            latest = pkg_latest.get(norm_name)
            if not latest:
                continue
            current = _extract_min_version(constraint)
            try:
                latest_ver = Version(latest)
            except Exception:
                continue
            if current is not None and current >= latest_ver:
                continue
            file_updates.append((orig_name, constraint or "any", latest))
        if file_updates:
            files_to_update[pyproject_path] = file_updates

    if not files_to_update:
        console.print("[green]✓ 所有依赖已是最新[/green]")
        return

    total_updates = sum(len(v) for v in files_to_update.values())
    console.print(f"[cyan]将更新 {total_updates} 个依赖包：[/cyan]")
    for pyproject_path, upds in files_to_update.items():
        rel = pyproject_path.relative_to(workspace_root)
        console.print(f"\n  [dim]{rel}:[/dim]")
        for name, old, new in upds:
            console.print(f"    {name}: {old} → [green]{new}[/green]")

    if dry_run:
        console.print("\n[yellow](dry run 模式，未实际修改文件)[/yellow]")
        return

    for pyproject_path, upds in files_to_update.items():
        content = pyproject_path.read_text(encoding="utf-8")
        for name, _, new_ver in upds:
            content, _ = _update_dep_in_content(content, name, new_ver)
        pyproject_path.write_text(content, encoding="utf-8")

    console.print(f"\n[green]✓ 已更新 {total_updates} 个依赖版本约束[/green]")

    if not no_install:
        _run_dependency_install(workspace_root)


def _run_dependency_install(workspace_root: Path) -> None:
    pdm_lock = workspace_root / "pdm.lock"
    uv_lock = workspace_root / "uv.lock"

    if pdm_lock.exists() or (workspace_root / ".pdm-python").exists():
        console.print("[cyan]检测到 PDM 环境，运行 pdm install...[/cyan]")
        cmd = [sys.executable, "-m", "pdm", "update", "--update-eager"]
    elif uv_lock.exists():
        console.print("[cyan]检测到 uv 环境，运行 uv lock --upgrade...[/cyan]")
        cmd = [sys.executable, "-m", "uv", "lock", "--upgrade"]
    else:
        console.print("[cyan]运行 pip install --upgrade...[/cyan]")
        cmd = [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "--upgrade"]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("正在安装更新后的依赖...", total=None)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=workspace_root)
            progress.update(task, completed=True)
        except FileNotFoundError as e:
            progress.update(task, completed=True)
            console.print(f"[yellow]命令不可用: {e}，请手动安装依赖[/yellow]")
            return

    if result.returncode == 0:
        console.print("[green]✓ 依赖安装成功[/green]")
    else:
        console.print("[yellow]⚠ 依赖安装可能有问题，请手动检查[/yellow]")
        if result.stderr:
            console.print(f"[dim]{result.stderr[:300]}[/dim]")


app = typer.Typer(help="依赖管理命令", add_completion=False)
app.command("check")(deps_check)
app.command("tree")(deps_tree)
app.command("outdated")(deps_outdated)
app.command("update")(deps_update)
