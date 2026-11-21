"""xs CLI 主应用装配模块（P1-3）。

命令不再由本模块逐个 import 并注册，而是装配内置插件 ``xs.commands``：
插件把命令登记到 :class:`~xs.plugins.registry.CommandRegistry`，本模块只负责
取出登记项并构建 Typer 应用。新增/移除内置命令只需改插件声明表，无需改本文件。
"""

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import cast

import typer
from rich.console import Console
from xuan_plugin.context import Context
from xuan_plugin.plugin import FiberState

from .plugins.commands import PLUGIN as COMMANDS_PLUGIN
from .plugins.registry import CommandEntry, CommandRegistry

# 版本号单一事实来源为 pyproject.toml，安装后由发行版元数据承载
try:
    __version__ = _dist_version("xs-cli")
except PackageNotFoundError:  # 未安装（如直接从源码树运行）时的回退
    __version__ = "0.0.0"

# Windows 控制台/CI（如 GitHub Actions windows runner）默认编码可能非 UTF-8，
# 输出中文时触发 UnicodeEncodeError。入口处统一强制 UTF-8 输出流。
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

console = Console()


def _version_callback(value: bool) -> None:
    """版本回调函数"""
    if value:
        console.print(f"[bold cyan]xs[/bold cyan] version [bold green]{__version__}[/bold green]")
        console.print("[dim]Xuanspace（玄境）monorepo 工具链[/dim]")
        raise typer.Exit()


def _register_commands(cli: typer.Typer, entries: tuple[CommandEntry, ...]) -> None:
    """按登记顺序把命令挂到根应用。"""
    for entry in entries:
        if entry.kind == "group":
            cli.add_typer(cast(typer.Typer, entry.target), name=entry.name)
        else:
            cli.command(entry.name)(entry.target)


def build_app() -> typer.Typer:
    """装配内置命令插件并构建根应用。"""
    cli = typer.Typer(
        name="xs",
        help="Xuanspace（玄境）monorepo 命令行工具",
        add_completion=False,
        no_args_is_help=True,
    )

    registry = CommandRegistry()
    ctx = Context()
    fiber = ctx.plugin(COMMANDS_PLUGIN, {"registry": registry})
    if fiber.state is not FiberState.ACTIVE:
        raise RuntimeError(f"内置命令插件未激活：{fiber.error!r}")

    _register_commands(cli, registry.entries())

    @cli.callback()
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

    return cli


app = build_app()


if __name__ == "__main__":
    app()
