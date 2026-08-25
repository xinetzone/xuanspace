"""
xs doctor 命令模块
检查开发环境配置和工具可用性
"""


import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import (
    check_python_version,
    find_workspace_root,
    get_python_version,
    get_python_version_requirement,
    load_pyproject,
)

console = Console()


@dataclass
class CheckResult:
    """检查结果数据类"""

    name: str
    status: str
    version: str | None = None
    message: str | None = None
    required: bool = True
    install_hint: str | None = None


def _run_command(cmd: list[str]) -> tuple[bool, str, str]:
    """
    运行外部命令并返回结果

    Args:
        cmd: 命令和参数列表

    Returns:
        (success, stdout, stderr) 元组
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, "", ""


def _get_version_from_output(output: str) -> str | None:
    """从命令输出中提取版本号"""
    import re

    match = re.search(r"(\d+\.\d+\.\d+)", output)
    if match:
        return match.group(1)
    return None


def check_python() -> CheckResult:
    """检查 Python 版本"""
    version = get_python_version()
    if check_python_version((3, 14)):
        return CheckResult(
            name="Python",
            status="ok",
            version=version,
            message="Python 版本满足要求",
        )
    else:
        return CheckResult(
            name="Python",
            status="error",
            version=version,
            message=f"Python 版本过低，需要 >= 3.14，当前为 {version}",
            install_hint="请安装 Python 3.14 或更高版本: https://www.python.org/downloads/",
        )


def check_git() -> CheckResult:
    """检查 Git 是否可用"""
    success, stdout, _ = _run_command(["git", "--version"])
    if success:
        version = _get_version_from_output(stdout)
        return CheckResult(
            name="Git",
            status="ok",
            version=version,
            message="Git 已安装",
        )
    else:
        return CheckResult(
            name="Git",
            status="warning",
            message="Git 未找到，版本控制功能不可用",
            required=False,
            install_hint="请安装 Git: https://git-scm.com/downloads",
        )


def check_package_manager() -> list[CheckResult]:
    """检查包管理器可用性"""
    results = []

    managers = [
        ("pdm", ["pdm", "--version"], "pip install pdm"),
        ("uv", ["uv", "--version"], "pip install uv"),
        ("pip", [sys.executable, "-m", "pip", "--version"], None),
    ]

    for name, cmd, install_hint in managers:
        success, stdout, _ = _run_command(cmd)
        if success:
            version = _get_version_from_output(stdout)
            results.append(
                CheckResult(
                    name=name,
                    status="ok",
                    version=version,
                    message=f"{name} 已安装",
                    required=(name == "pip"),
                )
            )
        else:
            results.append(
                CheckResult(
                    name=name,
                    status="warning" if name != "pip" else "error",
                    message=f"{name} 未找到",
                    required=(name == "pip"),
                    install_hint=f"请安装 {name}: {install_hint}" if install_hint else None,
                )
            )

    return results


def check_cmake() -> CheckResult:
    """检查 CMake 是否可用"""
    success, stdout, _ = _run_command(["cmake", "--version"])
    if success:
        version = _get_version_from_output(stdout)
        return CheckResult(
            name="CMake",
            status="ok",
            version=version,
            message="CMake 已安装",
            required=False,
        )
    else:
        return CheckResult(
            name="CMake",
            status="warning",
            message="CMake 未找到，无法构建 Native 项目",
            required=False,
            install_hint="请安装 CMake: https://cmake.org/download/ 或运行 xs toolchain install cmake",
        )


def check_ninja() -> CheckResult:
    """检查 Ninja 是否可用"""
    success, stdout, _ = _run_command(["ninja", "--version"])
    if success:
        version = stdout.split("\n")[0].strip() if stdout else None
        return CheckResult(
            name="Ninja",
            status="ok",
            version=version,
            message="Ninja 已安装",
            required=False,
        )
    else:
        return CheckResult(
            name="Ninja",
            status="warning",
            message="Ninja 未找到，Native 项目构建将使用默认生成器",
            required=False,
            install_hint="请安装 Ninja: https://ninja-build.org/ 或运行 xs toolchain install ninja",
        )


def check_sphinx(workspace_root: Path) -> CheckResult:
    """检查 Sphinx 是否可用（如果 docs/ 存在）"""
    docs_dir = workspace_root / "docs"
    if not docs_dir.exists():
        return CheckResult(
            name="Sphinx",
            status="skip",
            message="docs/ 目录不存在，跳过 Sphinx 检查",
            required=False,
        )

    success, stdout, _ = _run_command([sys.executable, "-m", "sphinx", "--version"])
    if success:
        version = _get_version_from_output(stdout)
        return CheckResult(
            name="Sphinx",
            status="ok",
            version=version,
            message="Sphinx 已安装",
            required=False,
        )
    else:
        return CheckResult(
            name="Sphinx",
            status="warning",
            message="Sphinx 未找到，文档构建功能不可用",
            required=False,
            install_hint="请安装 Sphinx: pip install sphinx myst-parser sphinx-book-theme",
        )


_PY_MINOR_REF = re.compile(r"\b3\.(\d+)(?:\.\d+)?\+")


def check_version_consistency(workspace_root: Path) -> CheckResult:
    """检查根文档中的 Python 版本声明是否与 pyproject.toml 的 requires-python 一致"""
    pyproject_path = workspace_root / "pyproject.toml"
    if not pyproject_path.exists():
        return CheckResult(
            name="版本一致性",
            status="skip",
            message="根 pyproject.toml 不存在，跳过版本一致性检查",
            required=False,
        )

    try:
        pyproject = load_pyproject(pyproject_path)
    except Exception:
        return CheckResult(
            name="版本一致性",
            status="warning",
            message="根 pyproject.toml 解析失败，跳过版本一致性检查",
            required=False,
        )

    requires_python = get_python_version_requirement(pyproject) or ""
    match = re.search(r"3\.(\d+)", requires_python)
    required_minor = int(match.group(1)) if match else None
    if required_minor is None:
        return CheckResult(
            name="版本一致性",
            status="skip",
            message="未在 pyproject.toml 声明 requires-python",
            required=False,
        )

    doc_files = ("CHANGELOG.md", "README.md", "AGENTS.md")
    mismatches: list[str] = []
    for fname in doc_files:
        path = workspace_root / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in _PY_MINOR_REF.finditer(text):
            if int(m.group(1)) < required_minor:
                mismatches.append(f"{fname}:{m.group(0)}")

    if mismatches:
        return CheckResult(
            name="版本一致性",
            status="error",
            version=requires_python,
            message=f"{len(mismatches)} 处 Python 版本声明低于 {requires_python}",
            install_hint="需同步: " + "; ".join(mismatches),
        )

    return CheckResult(
        name="版本一致性",
        status="ok",
        version=requires_python,
        message="CHANGELOG/README/AGENTS.md 版本声明与 pyproject.toml 一致",
    )


def _get_status_style(status: str) -> tuple[str, str]:
    """获取状态对应的图标和颜色"""
    styles = {
        "ok": ("✓", "green"),
        "warning": ("⚠", "yellow"),
        "error": ("✗", "red"),
        "skip": ("-", "dim"),
    }
    return styles.get(status, ("?", "white"))


def run_doctor(check_mode: bool = False) -> int:
    """
    运行 doctor 检查

    Args:
        check_mode: 如果为 True，在有必需工具缺失时返回非零退出码

    Returns:
        退出码：0 表示全部通过，1 表示有必需工具缺失
    """
    try:
        workspace_root = find_workspace_root()
    except RuntimeError:
        workspace_root = Path.cwd()

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Xuanspace 环境诊断[/bold cyan]\n工作区: [dim]{workspace_root}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    results: list[CheckResult] = []

    results.append(check_python())
    results.append(check_git())
    results.extend(check_package_manager())
    results.append(check_cmake())
    results.append(check_ninja())
    results.append(check_sphinx(workspace_root))
    results.append(check_version_consistency(workspace_root))

    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("状态", width=6, justify="center")
    table.add_column("工具", style="bold", width=12)
    table.add_column("版本", width=15)
    table.add_column("信息")

    has_errors = False
    has_warnings = False

    for result in results:
        icon, color = _get_status_style(result.status)
        if result.status == "error":
            has_errors = True
        elif result.status == "warning":
            has_warnings = True

        version_display = result.version or "-"
        message = result.message or ""

        table.add_row(
            f"[{color}]{icon}[/{color}]",
            result.name,
            version_display,
            f"[{color}]{message}[/{color}]" if result.status != "ok" else message,
        )

    console.print(table)
    console.print()

    hints = [r for r in results if r.install_hint and r.status in ("error", "warning")]
    if hints:
        console.print("[bold yellow]安装建议:[/bold yellow]")
        for hint in hints:
            console.print(f"  [dim]•[/dim] {hint.name}: {hint.install_hint}")
        console.print()

    if has_errors:
        console.print("[bold red]✗ 发现必需工具缺失，请安装后重试[/bold red]")
        return 1
    elif has_warnings:
        console.print("[bold yellow]⚠ 部分可选工具未安装，不影响核心功能[/bold yellow]")
    else:
        console.print("[bold green]✓ 所有检查通过！[/bold green]")

    return 0


def doctor() -> None:
    """运行环境诊断检查"""
    exit_code = run_doctor(check_mode=False)
    if exit_code != 0:
        raise typer.Exit(exit_code)
