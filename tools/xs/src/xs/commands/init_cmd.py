"""
xs init 命令模块
初始化 Xuanspace 工作区
"""


import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from ..config import check_python_version, find_workspace_root, get_python_version

console = Console()

WORKSPACE_GITIGNORE = """__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg
.eggs/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.venv/
venv/
env/
*.bat
!docs/make.bat
.temp/
external/
"""

WORKSPACE_PYPROJECT_TEMPLATE = """[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.1.0"
description = "Xuanspace monorepo 工作区"
requires-python = ">=3.13"
dependencies = []

[project.optional-dependencies]
docs = [
    "sphinx>=8.0",
    "myst-parser>=4.0",
    "sphinx-book-theme>=1.1",
    "sphinx-design",
    "sphinx-copybutton",
    "sphinxcontrib-mermaid",
]
test = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-xdist>=3.5",
]
lint = [
    "mypy>=1.10",
    "ruff>=0.4",
    "black>=24.4",
    "isort>=5.13",
]
build = [
    "build>=1.2",
    "scikit-build-core>=0.10",
    "cmake>=3.26",
    "ninja",
]
dev = [
    "pdm",
    "typer>=0.12",
    "rich>=13.0",
    "packaging>=24.0",
    "tomli-w>=1.0",
    "{name}[docs,test,lint,build]",
]

[tool.setuptools.packages.find]
where = ["libs"]

[tool.pdm.dev-dependencies]
dev = ["{name}[dev]"]
"""

WORKSPACE_README = """# {name}

Xuanspace（玄境）monorepo 工作区。

## 快速开始

```bash
# 环境检查
xs doctor

# 查看所有项目
xs list

# 创建新项目
xs new --type python my-lib
xs new --type python --app my-app
xs new --type native my-ext

# 构建
xs build

# 文档
xs docs build
xs docs serve
```

## 目录结构

- `apps/` — 应用项目
- `libs/` — 库项目
- `tools/` — 工具脚本和 CLI
- `docs/` — 项目文档
- `scripts/` — 维护脚本
- `vendor/` — 第三方依赖
- `attic/` — 归档项目
- `.meta/` — 文档元数据
- `.agents/` — AI 智能体规范
"""

DOCS_CONF = """project = "{name}"
copyright = "2024"
author = ""
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
"""

DOCS_INDEX = """# {name} 文档

欢迎使用 {name}！

## 目录

```{{toctree}}
:maxdepth: 2
:caption: 指南

quickstart
architecture
build-system
contributing
```

```{{toctree}}
:maxdepth: 1
:caption: CLI 参考

cli/index
```

```{{toctree}}
:maxdepth: 1
:caption: 其他

user-guide/index
```
"""

DOCS_QUICKSTART = """# 快速开始

## 安装

```bash
pip install -e ".[dev]"
```

## 基本使用

```bash
xs list
xs doctor
xs build
```
"""


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


def _create_file(path: Path, content: str, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        console.print(f"[dim]  跳过已存在: {path.name}[/dim]")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    console.print(f"[green]  ✓ 创建 {path.name}[/green]")
    return True


def _create_gitkeep(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    gitkeep = path / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def init_workspace(
    name: str | None = typer.Option(None, "--name", "-n", help="工作区名称，默认为当前目录名"),
    force: bool = typer.Option(False, "--force", "-f", help="覆盖已存在的文件"),
    scaffold: bool = typer.Option(False, "--scaffold", "-s", help="创建全新工作区脚手架（在空目录中使用）"),
) -> None:
    """初始化当前目录为 Xuanspace 工作区"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Xuanspace（玄境）工作区初始化[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    cwd = Path.cwd().resolve()
    pyproject = cwd / "pyproject.toml"

    is_existing = False
    if pyproject.exists():
        try:
            root = find_workspace_root(cwd)
            if root == cwd:
                is_existing = True
                console.print("[green]✓ 当前目录已是 Xuanspace 工作区[/green]")
                console.print(f"[dim]工作区根目录: {root}[/dim]")
            else:
                console.print(f"[yellow]⚠ 上级目录 {root} 是工作区根目录[/yellow]")
        except RuntimeError:
            pass

    console.print()
    console.print("[bold]环境检查:[/bold]")
    py_ok = _check_python()
    _check_git()
    console.print()

    if not py_ok:
        raise typer.Exit(1)

    if scaffold:
        ws_name = name or cwd.name
        console.print(f"[bold]创建工作区脚手架: {ws_name}[/bold]")
        console.print()

        _create_gitkeep(cwd / "apps")
        _create_gitkeep(cwd / "libs")
        _create_gitkeep(cwd / "tools")
        _create_gitkeep(cwd / "scripts")
        _create_gitkeep(cwd / "vendor")
        _create_gitkeep(cwd / "attic")
        _create_gitkeep(cwd / "docs" / "_static")
        _create_gitkeep(cwd / "docs" / "cli")
        _create_gitkeep(cwd / "docs" / "user-guide")
        _create_gitkeep(cwd / "tests")

        _create_file(cwd / ".gitignore", WORKSPACE_GITIGNORE, overwrite=force)
        _create_file(cwd / "pyproject.toml", WORKSPACE_PYPROJECT_TEMPLATE.format(name=ws_name), overwrite=force)
        _create_file(cwd / "README.md", WORKSPACE_README.format(name=ws_name), overwrite=force)
        _create_file(cwd / "docs" / "conf.py", DOCS_CONF.format(name=ws_name), overwrite=force)
        _create_file(cwd / "docs" / "index.md", DOCS_INDEX.format(name=ws_name), overwrite=force)
        _create_file(cwd / "docs" / "quickstart.md", DOCS_QUICKSTART, overwrite=force)

        console.print()
        console.print(
            Panel.fit(
                f"[bold green]✓ 工作区 {ws_name} 初始化完成！[/bold green]\n\n"
                "[bold]下一步:[/bold]\n"
                '  1. pip install -e ".[dev]"  安装开发依赖\n'
                "  2. xs doctor                检查环境\n"
                "  3. xs new --type python my-lib  创建第一个项目",
                border_style="green",
            )
        )
        return

    if not is_existing:
        console.print("[yellow]当前目录不是 Xuanspace 工作区[/yellow]")
        console.print("[dim]使用 --scaffold 参数创建全新工作区，或在 Xuanspace 仓库根目录运行[/dim]")
        console.print()

    console.print(
        Panel.fit(
            "[bold green]✓ 环境就绪！[/bold green]\n\n"
            "[bold]快速开始:[/bold]\n"
            "  xs list         查看所有项目\n"
            "  xs doctor       环境诊断\n"
            "  xs new --type python my-lib  创建新项目\n"
            "  xs build        构建项目\n"
            "  xs docs build   构建文档",
            border_style="green",
        )
    )
