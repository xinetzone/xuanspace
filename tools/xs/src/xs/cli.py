"""
xs CLI 主应用模块
使用 typer 组装所有子命令
"""

from __future__ import annotations

import typer
from rich.console import Console

from . import __version__
from .commands.build_cmd import build_project
from .commands.deps_cmd import app as deps_app
from .commands.docs_cmd import app as docs_app
from .commands.doctor_cmd import doctor
from .commands.init_cmd import init_workspace
from .commands.list_cmd import list_projects
from .commands.meta_cmd import app as meta_app
from .commands.new_cmd import create_new_project
from .commands.py_compat_cmd import py_compat
from .commands.toolchain_cmd import app as toolchain_app
from .commands.update_cmd import update_cmd
from .commands.version_cmd import app as version_app

console = Console()

app = typer.Typer(
    name="xs",
    help="Xuanspace（玄境）monorepo 命令行工具",
    add_completion=False,
    no_args_is_help=True,
)

app.add_typer(toolchain_app, name="toolchain")
app.add_typer(deps_app, name="deps")
app.add_typer(docs_app, name="docs")
app.add_typer(version_app, name="version")
app.add_typer(meta_app, name="meta")


def _version_callback(value: bool) -> None:
    """版本回调函数"""
    if value:
        console.print(f"[bold cyan]xs[/bold cyan] version [bold green]{__version__}[/bold green]")
        console.print(f"[dim]Xuanspace（玄境）monorepo 工具链[/dim]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="显示版本信息",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Xuanspace（玄境）monorepo 命令行工具

    提供项目管理、构建、环境检查等功能。
    """
    pass


app.command("list")(list_projects)
app.command("doctor")(doctor)
app.command("init")(init_workspace)
app.command("build")(build_project)
app.command("new")(create_new_project)
app.command("update")(update_cmd)
app.command("py-compat")(py_compat)


if __name__ == "__main__":
    app()
