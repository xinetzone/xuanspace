"""
xs py-compat 命令模块
检查 Python 包与 Python 3.13 的兼容性
"""

from __future__ import annotations

import json
import sys
import urllib.request
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

PYPI_URL = "https://pypi.org/pypi/{package}/json"
REQUIRED_PYTHON = (3, 13)


def _parse_version(version_str: str) -> tuple[int, ...]:
    import re
    nums = re.findall(r"\d+", version_str)
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def _check_python_requires(requires_python: str | None) -> tuple[bool, str]:
    if not requires_python:
        return True, "未指定（通常兼容）"
    if ">=" in requires_python:
        import re
        match = re.search(r">=\s*(\d+\.\d+)", requires_python)
        if match:
            min_ver = _parse_version(match.group(1))
            if len(min_ver) >= 2 and min_ver[:2] <= REQUIRED_PYTHON:
                return True, f"requires-python: {requires_python} ✓"
            return False, f"requires-python: {requires_python} ✗"
    if ">" in requires_python:
        import re
        match = re.search(r">\s*(\d+\.\d+)", requires_python)
        if match:
            min_ver = _parse_version(match.group(1))
            if len(min_ver) >= 2 and min_ver[:2] < REQUIRED_PYTHON:
                return True, f"requires-python: {requires_python} ✓"
            return False, f"requires-python: {requires_python} ✗"
    return True, f"requires-python: {requires_python}（需手动验证）"


def _fetch_package_info(package: str) -> dict | None:
    try:
        url = PYPI_URL.format(package=package)
        req = urllib.request.Request(url, headers={"User-Agent": "xuanspace-cli/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def py_compat(
    packages: list[str] = typer.Argument(None, help="要检查的包名（空格分隔），不传则检查当前项目依赖"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON 格式输出"),
) -> None:
    """检查 Python 包与 Python 3.13 的兼容性"""
    from ..config import find_workspace_root, get_python_version_requirement, load_pyproject
    from ..discovery import discover_projects

    if not packages:
        workspace_root = find_workspace_root()
        projects = discover_projects(workspace_root)
        dep_set: set[str] = set()
        try:
            root_data = load_pyproject(workspace_root / "pyproject.toml")
            for dep in root_data.get("project", {}).get("dependencies", []):
                import re as _re
                name = _re.split(r"[<>=!~;\[]", dep)[0].strip()
                if name:
                    dep_set.add(name)
            for dep in root_data.get("project", {}).get("optional-dependencies", {}).get("dev", []):
                name = _re.split(r"[<>=!~;\[]", dep)[0].strip()
                if name:
                    dep_set.add(name)
        except Exception:
            pass
        for proj in projects:
            for dep in proj.dependencies:
                dep_set.add(dep)
        packages = sorted(dep_set)
        if not packages:
            console.print("[yellow]未发现依赖包，请指定包名检查[/yellow]")
            return

    if json_output:
        results = {}
        for pkg in packages:
            info = _fetch_package_info(pkg)
            if info is None:
                results[pkg] = {"compatible": None, "reason": "无法获取包信息"}
                continue
            latest = info.get("info", {})
            requires = latest.get("requires_python")
            ok, reason = _check_python_requires(requires)
            results[pkg] = {"compatible": ok, "reason": reason, "requires_python": requires, "version": latest.get("version")}
        console.print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    table = Table(title=f"Python 3.13 兼容性检查", show_header=True, header_style="bold magenta")
    table.add_column("包名", style="cyan")
    table.add_column("最新版本", style="green", width=12)
    table.add_column("requires-python", width=20)
    table.add_column("兼容")
    table.add_column("说明")

    compatible = 0
    incompatible = 0
    unknown = 0

    with console.status("[cyan]正在查询 PyPI...[/cyan]"):
        for pkg in packages:
            info = _fetch_package_info(pkg)
            if info is None:
                table.add_row(pkg, "-", "-", "[yellow]?[/yellow]", "无法获取信息")
                unknown += 1
                continue
            latest = info.get("info", {})
            version = latest.get("version", "?")
            requires = latest.get("requires_python") or "-"
            ok, reason = _check_python_requires(latest.get("requires_python"))
            if ok:
                table.add_row(pkg, version, requires, "[green]✓[/green]", reason)
                compatible += 1
            else:
                table.add_row(pkg, version, requires, "[red]✗[/red]", reason)
                incompatible += 1

    console.print(table)
    console.print()
    if incompatible > 0:
        console.print(f"[yellow]⚠ {incompatible} 个包可能不兼容 Python 3.13，{compatible} 个兼容，{unknown} 个未知[/yellow]")
    else:
        console.print(f"[green]✓ 所有检查的包均兼容 Python 3.13 ({compatible} 兼容, {unknown} 未知)[/green]")
