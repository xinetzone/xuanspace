"""
xs docs 命令模块
文档构建和预览
"""


import subprocess
import sys
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import find_workspace_root

console = Console()


def _get_docs_dir(workspace_root: Path) -> Path:
    docs_dir = workspace_root / "docs"
    if not docs_dir.is_dir():
        console.print("[red]错误: docs/ 目录不存在[/red]")
        raise typer.Exit(1)
    return docs_dir


def docs_build(
    builder: str = typer.Option("html", "--builder", "-b", help="构建器类型: html, linkcheck"),
    output_dir: str | None = typer.Option(None, "--output", "-o", help="输出目录"),
) -> None:
    """构建 Sphinx 文档"""
    workspace_root = find_workspace_root()
    docs_dir = _get_docs_dir(workspace_root)

    out = output_dir or str(docs_dir / "_build" / builder)

    cmd = [sys.executable, "-m", "sphinx", "-b", builder, str(docs_dir), out]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"正在构建 {builder} 文档...", total=None)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=workspace_root)
            progress.update(task, completed=True)
        except FileNotFoundError:
            progress.update(task, completed=True)
            console.print("[red]错误: Sphinx 未安装，请运行 pip install sphinx myst-parser sphinx-book-theme[/red]")
            raise typer.Exit(1)

    if result.returncode == 0:
        console.print(f"[green]✓ 文档构建成功，输出目录: {out}[/green]")
        if builder == "html":
            index = Path(out) / "index.html"
            if index.exists():
                console.print(f"[dim]首页: {index}[/dim]")
    else:
        console.print("[red]✗ 文档构建失败[/red]")
        if result.stderr:
            for line in result.stderr.split("\n")[-20:]:
                if line.strip():
                    console.print(f"[red]{line}[/red]")
        raise typer.Exit(1)


def docs_serve(
    port: int = typer.Option(8000, "--port", "-p", help="HTTP 服务端口"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不自动打开浏览器"),
) -> None:
    """本地预览文档（启动 HTTP 服务）"""
    workspace_root = find_workspace_root()
    docs_dir = _get_docs_dir(workspace_root)
    build_dir = docs_dir / "_build" / "html"

    if not (build_dir / "index.html").exists():
        console.print("[yellow]文档尚未构建，正在先构建...[/yellow]")
        build_result = subprocess.run(
            [sys.executable, "-m", "sphinx", "-b", "html", str(docs_dir), str(build_dir)],
            capture_output=True,
            text=True,
            check=False,
            cwd=workspace_root,
        )
        if build_result.returncode != 0:
            console.print("[red]文档构建失败，请先运行 xs docs build[/red]")
            raise typer.Exit(1)

    url = f"http://localhost:{port}"
    console.print(f"[cyan]文档预览服务启动中: {url}[/cyan]")
    console.print("[dim]按 Ctrl+C 停止服务[/dim]")

    if not no_browser:
        webbrowser.open(url)

    try:
        import functools
        import http.server

        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler,
            directory=str(build_dir),
        )
        with http.server.HTTPServer(("localhost", port), handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[green]服务已停止[/green]")


def docs_clean() -> None:
    """清理文档构建产物"""
    import shutil

    workspace_root = find_workspace_root()
    build_dir = workspace_root / "docs" / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        console.print(f"[green]✓ 已清理: {build_dir}[/green]")
    else:
        console.print("[dim]没有需要清理的构建产物[/dim]")


def docs_linkcheck() -> None:
    """检查文档中的链接有效性"""
    docs_build(builder="linkcheck")


app = typer.Typer(help="文档管理命令", add_completion=False)
app.command("build")(docs_build)
app.command("serve")(docs_serve)
app.command("clean")(docs_clean)
app.command("linkcheck")(docs_linkcheck)
